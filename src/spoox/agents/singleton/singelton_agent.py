import asyncio
import re
from typing import List, Optional, Tuple

from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, AssistantMessage, \
    FunctionExecutionResultMessage, CreateResult

from spoox.agents.agent_system import AgentSystem
from spoox.agents.errors import ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, AgentError
from spoox.agents.singleton.messages import PublicMessage
from spoox.agents.singleton.prompts import get_SINGLETON_SYSTEM_PROMPT


# to ensure progress and prevent endless text-only replies, a limit on consecutive text-only messages is enforced
MAX_ONLY_TEXT_MESSAGES = 3

# if the model client encounters errors, the agent allows a maximum of three retry attempts
MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class SingletonAgent(RoutedAgent):
    """
    This agent represents the single-agent setup and is prompted to solve the entire task on its own. It is equipped
    with all available tools, and to provide additional guidance, we included a brief sequential abstraction
    of a typical problem-solving procedure consisting of the steps: explore, plan, solve, test, and summarize.

    As soon as it receives a RequestToSpeak, it starts working by calling tools and reasoning aloud
    until it finishes by including the finished_tag in its response.
    """

    finished_tag = "finished"

    def __init__(self, agent_system: AgentSystem, max_internal_iterations: int = 100) -> None:

        super().__init__(description="Single agent responsible for handling and completing the entire task.")
        self._max_internal_iterations = max_internal_iterations

        self._environment = agent_system.environment
        self._model_client = agent_system.model_client
        self._interface = agent_system.interface
        self._usage_stats = agent_system.usage_stats
        self._save_logs_f = agent_system.save_logs
        self._return_next_time_possible_event = agent_system.timeout_event
        self._tools = self._environment.get_tools(self)

        system_message = get_SINGLETON_SYSTEM_PROMPT(
            self.finished_tag, self._environment.get_additional_tool_descriptions(self))
        self._chat_history: List[LLMMessage] = [SystemMessage(content=system_message)]

        # logging
        self._interface.print_logging(system_message, f"logging - {self.id.type} - system_message")
        for t in self._tools:
            self._interface.print_logging(str(t.schema), f"logging - {self.id.type} - tool_schema")

    @message_handler
    async def handle_request_to_speak(self, message: PublicMessage, ctx: MessageContext) -> None:
        """Agent is requested to speak: internal execution loop is started."""

        try:
            self._chat_history.append(message.body)
            await self._agent_loop(ctx)
        except AgentError as e:
            self._interface.print_highlight(str(e), "Agent Error")
            self._usage_stats["agent_errors"].append(e)
        except Exception as e:
            self._interface.print_highlight(str(e), "Unexpected Error")
            self._usage_stats["agent_errors"].append(e)

    async def _agent_loop(self, ctx: MessageContext):
        """Request the LLM, process its response, and repeat until the agent has finished."""

        # tracking consecutive model client errors and LLM "only-text" responses
        counter_only_text_messages = 0
        model_client_errors = 0

        for i in range(1, self._max_internal_iterations + 1):

            # handling agent system timeout event
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
                self._interface.print_thought(llm_res.thought)

            # check if just text response
            if isinstance(content, str):
                self._interface.print(content, f"{self.id.type} - message")
                # check if `finished_tag` is included
                patter = rf"\[[^\]]*{re.escape(self.finished_tag)}[^\]]*\]"
                if re.search(patter, content, flags=re.IGNORECASE):
                    return
                # check if MAX_ONLY_TEXT_MESSAGES is reached
                counter_only_text_messages += 1
                if counter_only_text_messages > MAX_ONLY_TEXT_MESSAGES:
                    raise MaxOnlyTextMessagesError(self.id.type, MAX_ONLY_TEXT_MESSAGES)
                continue

            # check if tool calls (if it is not string it has to be a list of tool calls)
            assert isinstance(content, list) and all(isinstance(call, FunctionCall) for call in content)
            counter_only_text_messages = 0
            tool_results_message = await self._exec_tools(ctx, content)
            self._chat_history.append(tool_results_message)

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

    async def _exec_tools(self, ctx: MessageContext, calls: list[FunctionCall]) -> FunctionExecutionResultMessage:
        """Executes available tool calls."""

        tool_results = await asyncio.gather(
            *[
                self._environment.execute_tool_call(
                    self._tools, c, ctx.cancellation_token, self._interface, self._usage_stats, self.id.type
                )
                for c in calls
            ]
        )
        return FunctionExecutionResultMessage(content=tool_results)
