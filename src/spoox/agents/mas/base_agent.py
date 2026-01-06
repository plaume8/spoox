import asyncio
import copy
import re
import uuid
from typing import List, Tuple, Optional, Union

from autogen_core import RoutedAgent, message_handler, MessageContext, DefaultTopicId, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, UserMessage, AssistantMessage, \
    FunctionExecutionResultMessage, CreateResult

from spoox.agents.agent_system import AgentSystem
from spoox.agents.errors import ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, AgentError
from spoox.agents.mas.agents.prompts import get_AGENT_FAILED_GROUP_CHAT_MESSAGE
from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak, GROUP_CHAT_TOPIC_TYPE


# to ensure progress and prevent endless text-only replies, a limit on consecutive text-only messages is enforced
MAX_ONLY_TEXT_MESSAGES = 3

# if the model client encounters errors, the agent allows a maximum of three retry attempts
MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class BaseGroupChatAgent(RoutedAgent):
    """
    Base agent class used to build agents that follow the concepts and design principles of the spoox framework.
    Agents maintain a local chat history by recording all distributed GroupChatMessages.
    When a RequestToSpeak is received, the agent starts its internal execution loop.
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
        Base agent class used to build agents following the concepts and design principles of the spoox framework.

        Args:
            description (str): One-sentence agent description passed to AutoGen's RoutedAgent.
            system_message (str): System message added as the initial message to the agent's message history.
            agent_system (AgentSystem): Agent system associated with the agent, providing access to the environment, model client, and other shared components.
            next_agent_topic_types (list[str]): List of all possible next agent topic types that the agent is allowed to call.
            max_internal_iterations (int): Maximum number of internal iterations the agent may perform, corresponding to the maximum number of LLM calls.
            fallback_agent_topic_type (str): Topic type of the agent to be invoked if this agent fails.
            reset_on_request_to_speak (bool): If True, internal messages are cleared from the chat history each time the agent is called, while group chat messages remain.
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
        Thereby, `_chat_history` stores all group chat messages as well as all internal message history
        and `_chat_history_group_chat_only` only tracks GroupChatMessages.
        This ensures that when the chat history is reset (controlled by `reset_on_request_to_speak`),
        the chat history is simply replaced with `_chat_history_group_chat_only`,
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

        # additional chat history logging
        logging_chat_hist = ' -> '.join([f"[{str(h.content)[:60].replace('\n', '')}…]" for h in self._chat_history])
        self._interface.print_logging('[start] -> ' + logging_chat_hist, f"logging - {self.id.type} - chat history")

        # run the agent's internal loop;
        # if agent loop fails, no final group chat message is generated, and the fallback agent is called if available
        try:
            await self._agent_loop(ctx)
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
            self._interface.print_shadow(failure_message, "Fallback Agent Call")
            await self._send_group_chat_message_and_request_to_speak(failure_message, self._fallback_agent_topic_type)

        # if error and no fallback -> just return -> no next agent will be triggered -> autogen runtime exits

    async def _agent_loop(self, ctx: MessageContext):
        """
        Request the LLM, process its response, and repeat until the agent has finished.
        High-level flow:

        1.	The model client (LLM) is queried and returns a response.
        2.	The response is checked for tool calls. If tool calls are present, the corresponding tools are executed and the loop restarts.
        3.	If the response consists of a plain text answer, it is checked for a requested next-agent tag and whether the referenced agent exists.
        """

        # tracking consecutive model client errors and LLM "only-text" responses
        counter_only_text_messages = 0
        model_client_errors = 0

        for i in range(1, self._max_internal_iterations + 1):

            # checking agent system timeout event
            if self._return_next_time_possible_event.is_set():
                return

            # logging
            self._save_logs_f()
            self._usage_stats['llm_calls_count'] += 1

            # request model client (llm)
            llm_res, model_client_errors = await self._request_llm(ctx, model_client_errors)
            if llm_res is None:
                continue
            self._interface.print_logging(str(llm_res), f"logging - {self.id.type} - entire llm_res")
            content = llm_res.content

            # add the response to session and print thoughts if available
            self._chat_history.append(
                AssistantMessage(content=content, thought=llm_res.thought, source=self.id.type))
            if llm_res.thought:
                self._interface.print_thought(llm_res.thought, f"{self.id.type} - thoughts")

            # check if list of tool calls
            tool_results_message = await self._exec_tools(ctx, content)
            if tool_results_message is not None:
                counter_only_text_messages = 0
                self._chat_history.append(tool_results_message)
                # trigger LLM again with tool results in _chat_history
                continue

            # check if just text (autogen: if it is not a list of tool calls, it has to be string)
            assert isinstance(content, str)
            self._interface.print(content, f"{self.id.type} - message")

            # check if agent finished and calls next agent
            next_agent_tag = self._includes_agent_tag(content)
            if next_agent_tag is not None:
                # we assume that if an agent tag is included, this message contains the summary for the group chat
                await self._send_group_chat_message_and_request_to_speak(content, next_agent_tag)
                return

            # check if no `_next_agent_topic_types` were defined; if so, the agent finishes if no tools are called
            if not self._next_agent_topic_types:
                await self._send_group_chat_message(content)
                return

            # check if MAX_ONLY_TEXT_MESSAGES is reached
            counter_only_text_messages += 1
            if counter_only_text_messages > MAX_ONLY_TEXT_MESSAGES:
                raise MaxOnlyTextMessagesError(self.id.type, MAX_ONLY_TEXT_MESSAGES)

        raise MaxIterationsError(self.id.type, self._max_internal_iterations)

    async def _request_llm(self, ctx: MessageContext, model_client_errors: int) -> Tuple[Optional[CreateResult], int]:
        """Invokes the model client (LLM) and handles any exception that occurs."""

        try:
            llm_res = await self._model_client.create(
                messages=self._chat_history,
                tools=self._tools,
                cancellation_token=ctx.cancellation_token,
            )
        except Exception as e:
            self._usage_stats['model_client_exceptions'].append(e)
            self._interface.print_logging(str(e), "Model Client Error")
            if model_client_errors >= MAX_MODEL_CLIENT_ERRORS_RETRIALS:
                raise ModelClientError(self.id.type, MAX_MODEL_CLIENT_ERRORS_RETRIALS, e)
            else:
                return None, model_client_errors + 1
        # llm call success
        self._usage_stats['prompt_tokens'].append(llm_res.usage.prompt_tokens)
        self._usage_stats['completion_tokens'].append(llm_res.usage.completion_tokens)
        return llm_res, 0

    async def _exec_tools(self, ctx: MessageContext, content: Union[str, List[FunctionCall]]) -> Optional[FunctionExecutionResultMessage]:
        """Executes available tool calls, if any."""

        # check whether the LLM response contains tool calls
        if not isinstance(content, list) or not all(isinstance(c, FunctionCall) for c in content):
            return None
        if self._environment is None:
            return None

        # execute tool calls
        tool_results = await asyncio.gather(
            *[
                self._environment.execute_tool_call(
                    self._tools, c, ctx.cancellation_token, self._interface, self._usage_stats, self.id.type
                )
                for c in content
            ]
        )
        return FunctionExecutionResultMessage(content=tool_results)

    def _includes_agent_tag(self, message: str) -> Optional[str]:
        """Checks if the message includes an agent tag and returns the first match detected."""

        for nt in self._next_agent_topic_types:
            patter = rf"\[[^\]]*{re.escape(nt)}[^\]]*\]"
            if re.search(patter, message, flags=re.IGNORECASE):
                self._usage_stats['next_agent_calling_chain'].append(nt)
                return nt
        return None

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

    async def _send_group_chat_message_and_request_to_speak(self, message: str, agent_type: str):
        await self._send_group_chat_message(message)
        # 0.1 delay to ensure the GroupChatMessage can be observed before the RequestToSpeak
        # (I think it is not required, however, it certainly does not hurt)
        await asyncio.sleep(0.1)
        await self._send_request_to_speak(agent_type)
