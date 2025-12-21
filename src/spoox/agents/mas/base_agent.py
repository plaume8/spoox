import asyncio
import re
import time
import uuid
import copy
from typing import List

from autogen_core import RoutedAgent, message_handler, MessageContext, DefaultTopicId, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, UserMessage, AssistantMessage, \
    FunctionExecutionResultMessage
from ollama import ResponseError

from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak, GROUP_CHAT_TOPIC_TYPE
from spoox.agents.mas.StructuredFlow.agents.prompts import get_AGENT_FAILED_GROUP_CHAT_MESSAGE
from spoox.agents.errors import MaxOllamaRetrialsError, ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, \
    AgentError

# just in case the model is not using the tools or calling a next agent and only responses with a text
# we have to make sure that there is a limit of "only text messages"
MAX_ONLY_TEXT_MESSAGES = 3

MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS = 5

MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class BaseGroupChatAgent(RoutedAgent):
    """
    Base agent class used to build agents that follow the concepts and design principles of the spoox framework.
    Tracks all distributed GroupChatMessages and adds them to the local chat history.
    If a RequestToSpeak is received the agent gets to work in a loop starting with requesting the ModelClient.


    """

    def __init__(
            self,
            description: str,
            system_message: str,
            agent_system: AgentSystem,
            next_agent_topic_types: list[str] = None,
            max_internal_iterations: int = 50,
            fallback_agent_topic_type: str = None,
            reset_on_request_to_speak: bool = False,  # todo should be True ?
    ) -> None:
        """
        Base agent class used to build agents that follow the concepts and design principles of the spoox framework.

        :param description (str): one-sentence agent description passed to the RoutedAgent.
        :param system_message (str): system message, added as the initial message to the agent's message history.
        :param agent_system (AgentSystem): agent system associated with the agent,
        providing access to the environment, model client, and other shared components.
        :param next_agent_topic_types (list[str]): list of all possible next agent topic types that the agent is allowed to call.
        :param max_internal_iterations (int): the maximum number of internal iterations the agent may perform,
        corresponding to the maximum number of LLM calls.
        :param fallback_agent_topic_type (str): topic type of the agent to be invoked if this agent fails.
        :param reset_on_request_to_speak (bool): if set to True, internal messages are cleared from the chat history each time the agent is called,
        while group chat messages always remain.
        """

        super().__init__(description=description)
        self._next_agent_topic_types = [n.lower() for n in next_agent_topic_types or []]
        self._max_internal_iterations = max_internal_iterations
        self._fallback_agent_topic_type = fallback_agent_topic_type
        self._reset_on_request_to_speak = reset_on_request_to_speak

        self._environment = agent_system.environment
        self._model_client = agent_system.model_client
        self._interface = agent_system.interface
        self._usage_stats = agent_system.usage_stats
        self._save_logs_f = agent_system.save_logs
        self._return_next_time_possible_event = agent_system.timeout_event
        self._tools = self._environment.get_tools(self) if self._environment else []

        self._chat_history: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._chat_history_group_chat_only: List[LLMMessage] = [SystemMessage(content=system_message)]

        # logging
        self._interface.print_logging(system_message, f"logging - {self.id.type} - system_message")
        for t in self._tools:
            self._interface.print_logging(str(t.schema), f"logging - {self.id.type} - tool_schema")

    @message_handler
    async def handle_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        """
        Each agent keeps track of the entire group chat in its internal message history.
        Therefore, it stores every incoming GroupChatMessage and tracks which agent posted each message.
        Thereby, `_chat_history` stores all group chat messages as well as all internal message history.
        In contrast, `_chat_history_group_chat_only` only tracks GroupChatMessages.
        This mechanism ensures that when the chat history is reset (controlled by `reset_on_request_to_speak`),
        the chat history is replaced with `_chat_history_group_chat_only`,
        so that only group chat messages are retained and all internal iteration messages are discarded.
        """
        new_messages = [
            UserMessage(content=f"Transferred to {message.body.source.capitalize()} agent.", source="system"),
            message.body,
        ]
        self._chat_history_group_chat_only.extend(new_messages)
        if message.body.source != self.id.type:
            self._chat_history.extend(new_messages)

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """
        Agent is requested to speak: parts of its internal state are reset, and the internal execution loop is started.
        Furthermore, if the agent loop throws errors, they are caught and logged, and a fallback mechanism is triggered.
        """

        # ensures the env is fully reset to prevent any influence from previous agents that used the same env
        if self._environment:
            await self._environment.reset()

        # reset chat history to group chat messages only; all previous internal iteration messages are discarded
        if self._reset_on_request_to_speak:
            self._chat_history = copy.deepcopy(self._chat_history_group_chat_only)
            self._interface.print_logging(
                "reset to group chat history only on request to speak", f"logging - {self.id.type} - reset")

        # add a system message that instructs the model to adopt this agent's persona
        self._chat_history.append(
            UserMessage(
                content=f"Transferred to {self.id.type.capitalize()} agent, adopt the persona immediately.",
                source="system"
            )
        )

        # logging  # todo simply remove these two lines after final testing
        logging_chat_hist = '| -> ' + ' -> '.join([str(h.content)[:40].replace('\n', '') for h in self._chat_history])
        self._interface.print_logging(logging_chat_hist, f"logging - {self.id.type} - chat history")

        # run the agent's internal loop;
        # if agent loop fails, no final group chat message is generated, and the fallback agent is called if available
        try:
            await self.agent_loop(ctx)
            return
        except AgentError as e:
            self._interface.print_highlight(str(e), "Agent Error")
            self._usage_stats["agent_errors"].append(e)
        except Exception as e:
            self._interface.print_highlight(str(e), "Unexpected Error")
            self._usage_stats["agent_errors"].append(e)

        # fallback mechanism
        if self._fallback_agent_topic_type:
            failure_message = get_AGENT_FAILED_GROUP_CHAT_MESSAGE(self.id.type, self._fallback_agent_topic_type)
            self._interface.print_shadow(failure_message)
            await self._send_group_chat_message(failure_message)
            # 0.1 delay to ensure the GroupChatMessage can be observed before the RequestToSpeak
            # (I think it is not required, however, it certainly does not hurt)
            await asyncio.sleep(0.1)
            await self._send_request_to_speak(self._fallback_agent_topic_type)

        # if error and no fallback -> just return -> no next agent will be triggered -> autogen runtime exits

    async def agent_loop(self, ctx: MessageContext):
        """Run llm over and over again until the agent is finished."""
        """
        if the agent is requested to speak, the llm is triggered;
        if the response includes the 'finished_tag' and no tool calls, the answer is printed and the agent exits;
        if the response contains tool calls, the tools are executed, and the llm is triggered again with the results;
        one of the tool calls could call a next agent, if so a RequestToSpeak message is posted.
        """

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
                    self._interface.print_shadow(
                        "Model client error -> retry (MAX_MODEL_CLIENT_ERRORS_RETRIALS not yet reached)",
                        "Model Client Error")
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
            topic_id=DefaultTopicId(type=GROUP_CHAT_TOPIC_TYPE),
        )

    async def _send_request_to_speak(self, agent_type: str):
        await self.publish_message(
            RequestToSpeak(nonce=str(uuid.uuid4())), DefaultTopicId(type=agent_type)
        )
