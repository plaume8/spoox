import asyncio
import time
from typing import List, Callable, Optional, Tuple

from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, ChatCompletionClient, AssistantMessage, \
    FunctionExecutionResultMessage, CreateResult

from spoox.agents.agent_system import AgentSystem
from spoox.agents.singleton.messages import PublicMessage
from spoox.agents.errors import ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, \
    AgentError
from spoox.environment.Environment import Environment
from spoox.interface import Interface
from spoox.agents.singleton.prompts import get_SINGLETON_SYSTEM_PROMPT

# just in case the model is not using the tools or including the finished_tag
# we have to make sure that there is a limit of "only text messages"
MAX_ONLY_TEXT_MESSAGES = 3

MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class SingletonAgent(RoutedAgent):
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
            await self.agent_loop(ctx)
        except AgentError as e:
            self._interface.print_highlight(str(e), "Agent Error")
            self._usage_stats["agent_errors"].append(e)
        except Exception as e:
            self._interface.print_highlight(str(e), "Unexpected Error")
            self._usage_stats["agent_errors"].append(e)

    async def agent_loop(self, ctx: MessageContext):
        """Run llm over and over again until the agent is finished."""

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

            # request llm
            llm_res, model_client_errors = self._request_llm(ctx, model_client_errors)
            if llm_res is None:
                continue

            # add the response to session
            self._chat_history.append(
                AssistantMessage(content=llm_res.content, thought=llm_res.thought, source=self.id.type))
            self._interface.print_logging(str(llm_res), f"logging - {self.id.type} - entire llm_res")

            # print thoughts if available
            if llm_res.thought:
                self._interface.print_thought(llm_res.thought, f"{self.id.type} - thought field")

            # check if just text
            if isinstance(llm_res.content, str):
                self._interface.print(llm_res.content, f"{self.id.type} - message")
                # check if `finished_tag` is included
                if f"[{self.finished_tag.lower()}]" in llm_res.content.lower():
                    return
                # check if MAX_ONLY_TEXT_MESSAGES is reached
                counter_only_text_messages += 1
                if counter_only_text_messages > MAX_ONLY_TEXT_MESSAGES:
                    raise MaxOnlyTextMessagesError(self.id.type, MAX_ONLY_TEXT_MESSAGES)
                continue

            # check if tool calls (if it is not string it has to be a list of tool calls)
            assert isinstance(llm_res.content, list) and all(
                isinstance(call, FunctionCall) for call in llm_res.content
            )
            counter_only_text_messages = 0
            # execute all tool calls and add results to session
            tool_results = await asyncio.gather(
                *[self._environment.execute_tool_call(self._tools, call, ctx.cancellation_token, self._interface,
                                                      self._usage_stats, self.id.type)
                  for call in llm_res.content]
            )
            self._chat_history.append(FunctionExecutionResultMessage(content=tool_results))

        raise MaxIterationsError(self.id.type, self._max_internal_iterations)

    async def _request_llm(self, ctx: MessageContext, model_client_errors: int):
        """Invokes the model client (LLM) and handles any exception that occurs."""

        try:
            llm_res = await self._model_client.create(
                messages=self._chat_history,
                tools=self._tools,
                cancellation_token=ctx.cancellation_token,
            )
        except Exception as e:
            self._usage_stats['model_client_exceptions'].append(e)
            self._interface.print_highlight(str(e), "Model Client Error (no retry)")
            if model_client_errors >= MAX_MODEL_CLIENT_ERRORS_RETRIALS:
                raise ModelClientError(self.id.type, MAX_MODEL_CLIENT_ERRORS_RETRIALS, e)
            else:
                self._interface.print_shadow(f"{str(e)}", "Model Client Error (retry)")
                return None, model_client_errors + 1
        # llm call success
        self._usage_stats['prompt_tokens'].append(llm_res.usage.prompt_tokens)
        self._usage_stats['completion_tokens'].append(llm_res.usage.completion_tokens)
        return CreateResult, 0
