import asyncio
import re
import time
import uuid
import copy
from typing import List, Callable

from autogen_core import RoutedAgent, message_handler, MessageContext, DefaultTopicId, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, UserMessage, ChatCompletionClient, AssistantMessage, \
    FunctionExecutionResultMessage
from ollama import ResponseError

from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak
from spoox.agents.mas.StructuredFlow.agents.utils import get_AGENT_FAILED_GROUP_CHAT_MESSAGE
from spoox.agents.errors import MaxOllamaRetrialsError, ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, \
    AgentError
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface

# just in case the model is not using the tools or calling a next agent and only responses with a text
# we have to make sure that there is a limit of "only text messages"
MAX_ONLY_TEXT_MESSAGES = 3

MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS = 5

MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class BaseGroupChatAgent(RoutedAgent):

    def __init__(
            self,
            group_chat_topic_type: str,
            description: str,
            system_message: str,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            return_next_time_possible_event: asyncio.Event,
            environment: Environment = None,
            next_agent_topic_types: [str] = None,
            max_internal_iterations: int = 50,
            fallback_agent_topic_type: str = None,
            reset_on_request_to_speak: bool = False,
    ) -> None:

        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type.lower()
        self._environment = environment
        self._model_client = model_client
        self._interface = interface
        self._usage_stats = usage_stats
        self._save_logs_f = save_logs_f
        self._next_agent_topic_types = [n.lower() for n in next_agent_topic_types or []]
        self._max_internal_iterations = max_internal_iterations
        self._fallback_agent_topic_type = fallback_agent_topic_type.lower() if fallback_agent_topic_type else None
        self._return_next_time_possible_event = return_next_time_possible_event
        self._reset_on_request_to_speak = reset_on_request_to_speak

        self._chat_history: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._chat_history_group_chat_only: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._tools = environment.get_tools(self) if self._environment else []

        for t in self._tools:
            self._interface.print_logging(str(t.schema), f"logging - {self.id.type} - tool_schema")
        self._interface.print_logging(system_message, f"logging - {self.id.type} - system_message")

    @message_handler
    async def handle_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        """
        each agent keeps its own group chat history;
        therefore, it has to process every GroupChatMessage and keeps track of whom posted the message;
        own group chat messages are only stored in the _chat_history_group_chat_only, such that if the chat history
        is reset (if reset_on_request_to_speak), the fallback chat history does also contain own previous summaries.
        """
        self._chat_history_group_chat_only.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source.capitalize()} agent.", source="system"),
                message.body,
            ]
        )
        if message.body.source != self.id.type:
            self._chat_history.extend(
                [
                    UserMessage(content=f"Transferred to {message.body.source.capitalize()} agent.", source="system"),
                    message.body,
                ]
            )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """
        if the agent is requested to speak, the llm is triggered;
        after responding, the next selected agent is called by posting a new RequestToSpeak message.
        """

        # make sure the environment is reset (no influence of previous agents)
        if self._environment:
            await self._environment.reset()

        # reset chat history to group chat history only
        if self._reset_on_request_to_speak:
            self._chat_history = copy.deepcopy(self._chat_history_group_chat_only)
            self._interface.print_logging("reset to group chat history only on request to speak",
                                          f"logging - {self.id.type} - reset")

        # add to agents own history -> system message to adopt persona now
        self._chat_history.append(
            UserMessage(
                content=f"Transferred to {self.id.type.capitalize()} agent, adopt the persona immediately.",
                source="system"
            )
        )

        # logging
        logging_chat_hist = '| -> ' + ' -> '.join([str(h.content)[:40].replace('\n', '') for h in self._chat_history])
        self._interface.print_logging(logging_chat_hist, f"logging - {self.id.type} - chat history")

        # run agent loop; if agent loop fails, the agent simply not generates any response
        try:
            await self.agent_loop(ctx)
            return
        except AgentError as e:
            self._interface.print_highlight(str(e), "Agent Error")
            self._usage_stats["agent_errors"].append((type(e).__name__, str(e)))
        except Exception as e:
            self._interface.print_highlight(str(e), "Unexpected Error")
            self._usage_stats["agent_errors"].append((type(e).__name__, str(e)))

        # check if fallback
        if self._fallback_agent_topic_type:
            failure_message = get_AGENT_FAILED_GROUP_CHAT_MESSAGE(self.id.type, self._fallback_agent_topic_type)
            self._interface.print_shadow(failure_message)
            await self._send_group_chat_message(failure_message)
            await asyncio.sleep(
                0.1)  # ensuring the group msg can be observed before the RTS (I think it is not required - but not sure...)
            await self._send_request_to_speak(self._fallback_agent_topic_type)

        # if error and no fallback just return -> no next agent will be triggered -> autogen runtime exits

    async def agent_loop(self, ctx: MessageContext):
        """Run llm over and over again until the agent is finished."""

        counter_only_text_messages = 0
        ollama_response_errors = 0
        model_client_errors = 0
        for i in range(1, self._max_internal_iterations + 1):

            # handling agent system timeout event
            if self._return_next_time_possible_event.is_set():
                return

            # request llm
            self._save_logs_f()
            self._usage_stats['llm_calls_count'] += 1
            try:
                llm_res = await self._model_client.create(
                    messages=self._chat_history,
                    tools=self._tools,
                    cancellation_token=ctx.cancellation_token,
                )
            except ResponseError as e:
                ollama_response_errors += 1
                self._usage_stats['ollama_response_error_count'] += 1  # todo remove
                self._interface.print_highlight(str(e), "Ollama ResponseError")
                if ollama_response_errors > MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS:
                    raise MaxOllamaRetrialsError(self.id.type, MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS)
                else:
                    self._interface.print_shadow(
                        "Ollama ResponseError -> retry (MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS not yet reached)",
                        "Ollama ResponseError")
                    continue
            except Exception as e:
                model_client_errors += 1
                self._usage_stats['model_client_exceptions'].append(str(e))
                self._interface.print_highlight(str(e), "Model Client Error")
                if model_client_errors > MAX_MODEL_CLIENT_ERRORS_RETRIALS:
                    raise ModelClientError(self.id.type, MAX_MODEL_CLIENT_ERRORS_RETRIALS, e)
                else:
                    self._interface.print_shadow("Model client error -> retry (MAX_MODEL_CLIENT_ERRORS_RETRIALS not yet reached)", "Model Client Error")
                    start_time = time.time()
                    await asyncio.sleep(60)
                    print(f"short delay ofter model client exception: {(time.time() - start_time) / 60}")
                    continue
            self._usage_stats['prompt_tokens'].append(llm_res.usage.prompt_tokens)
            self._usage_stats['completion_tokens'].append(llm_res.usage.completion_tokens)
            ollama_response_errors = 0
            model_client_errors = 0

            # add the response to session
            self._chat_history.append(
                AssistantMessage(content=llm_res.content, thought=llm_res.thought, source=self.id.type))
            self._interface.print_logging(str(llm_res), f"logging - {self.id.type} - entire llm_res")

            # print thoughts if available
            if llm_res.thought:
                self._interface.print_thought(llm_res.thought, f"{self.id.type} - thought field")

            # check if list of tool calls
            if isinstance(llm_res.content, list) and all(
                    isinstance(call, FunctionCall) for call in llm_res.content) and self._environment:
                # execute all tool calls and add results to session
                counter_only_text_messages = 0
                tool_results = await asyncio.gather(
                    *[self._environment.execute_tool_call(self._tools, call, ctx.cancellation_token, self._interface,
                                                          self._usage_stats, self.id.type)
                      for call in llm_res.content]
                )
                self._chat_history.append(FunctionExecutionResultMessage(content=tool_results))
                # SMASSupervisorAgent special case: if tools contained an AgentCall execution -> exit
                if any(call.name == "CallAgent" for call in llm_res.content):
                    # special logging for SMAS agent system
                    call_agent_tool_calls = [t.arguments for t in llm_res.content if t.name == "CallAgent"]
                    self._usage_stats['supervisor_agent_calling_chain'].append(call_agent_tool_calls)
                    return
                # otherwise: trigger LLM again with tool results in _chat_history
                continue

            # check if just text (autogen: if it is not a list of tool calls, it has to be string)
            assert isinstance(llm_res.content, str)
            self._interface.print(llm_res.content, f"{self.id.type} - message")

            # check if agent finished and calls next agent
            for nt in self._next_agent_topic_types:
                patter = rf"\[[^\]]*{re.escape(nt)}[^\]]*\]"
                if re.search(patter, llm_res.content, flags=re.IGNORECASE):
                    # logging
                    self._usage_stats['next_agent_calling_chain'].append(nt)
                    # we assume that if an agent tag is included, this message contains the summary for the group chat
                    await self._send_group_chat_message(llm_res.content)
                    await asyncio.sleep(
                        0.1)  # ensuring the group msg can be observed before the RTS (I think it is not required - but not sure...)
                    await self._send_request_to_speak(nt)
                    return

            # check if no `_next_agent_topic_types` were defined; if so, the agent finishes if no tools are called
            if not self._next_agent_topic_types:
                await self._send_group_chat_message(llm_res.content)
                return

            # check if MAX_ONLY_TEXT_MESSAGES is reached
            counter_only_text_messages += 1
            if counter_only_text_messages > MAX_ONLY_TEXT_MESSAGES:
                raise MaxOnlyTextMessagesError(self.id.type, MAX_ONLY_TEXT_MESSAGES)

        raise MaxIterationsError(self.id.type, self._max_internal_iterations)

    async def _send_group_chat_message(self, message: str):
        self._usage_stats['group_chat_message_lengths'].append(len(message))
        await self.publish_message(
            message=GroupChatMessage(
                nonce=str(uuid.uuid4()),
                body=UserMessage(content=message, source=self.id.type)
            ),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )

    async def _send_request_to_speak(self, agent_type: str):
        await self.publish_message(RequestToSpeak(nonce=str(uuid.uuid4())), DefaultTopicId(type=agent_type))
