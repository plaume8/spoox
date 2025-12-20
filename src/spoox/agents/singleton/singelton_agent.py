import asyncio
import time
from typing import List, Callable

from autogen_core import RoutedAgent, message_handler, MessageContext, FunctionCall
from autogen_core.models import SystemMessage, LLMMessage, ChatCompletionClient, AssistantMessage, \
    FunctionExecutionResultMessage
from ollama import ResponseError

from spoox.agents.singleton.messages import PublicMessage
from spoox.agents.errors import MaxOllamaRetrialsError, ModelClientError, MaxOnlyTextMessagesError, MaxIterationsError, \
    AgentError
from spoox.environment.Environment import Environment
from spoox.interface import Interface
from spoox.agents.singleton.prompts import get_SINGLETON_SYSTEM_PROMPT

# just in case the model is not using the tools or including the finished_tag
# we have to make sure that there is a limit of "only text messages"
MAX_ONLY_TEXT_MESSAGES = 3

MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS = 5

MAX_MODEL_CLIENT_ERRORS_RETRIALS = 3


class SingletonAgent(RoutedAgent):

    finished_tag = "finished"

    def __init__(
            self,
            environment: Environment,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            return_next_time_possible_event: asyncio.Event,
            max_internal_iterations: int = 100,
    ) -> None:

        super().__init__(description="Single agent responsible for handling and completing the entire task.")
        self._environment = environment
        self._model_client = model_client
        self._interface = interface
        self._usage_stats = usage_stats
        self._save_logs_f = save_logs_f
        self._return_next_time_possible_event = return_next_time_possible_event
        self._max_internal_iterations = max_internal_iterations

        system_message = SystemMessage(content=get_SINGLETON_SYSTEM_PROMPT(self.finished_tag, environment.get_additional_tool_descriptions(self)))
        self._chat_history: List[LLMMessage] = [system_message]
        self._tools = environment.get_tools(self)

        for t in self._tools:
            self._interface.print_logging(str(t.schema), f"logging - {self.id.type} - tool_schema")
        self._interface.print_logging(system_message.content, f"logging - {self.id.type} - system_message")

    @message_handler
    async def handle_request_to_speak(self, message: PublicMessage, ctx: MessageContext) -> None:
        """
        if the agent is requested to speak, the llm is triggered;
        if the response includes the 'finished_tag' and no tool calls, the answer is printed and the agent exits;
        if the response contains tool calls, the tools are executed, and the llm is triggered again with the results;
        one of the tool calls could call a next agent, if so a RequestToSpeak message is posted.
        """

        # add the user message to session
        self._chat_history.append(message.body)

        # run agent loop; if agent loop fails, the agent simply not generates any response
        try:
            await self.agent_loop(ctx)
        except AgentError as e:
            self._interface.print_highlight(str(e), "Agent Error")
            self._usage_stats["agent_errors"].append((type(e).__name__, str(e)))
        except Exception as e:
            self._interface.print_highlight(str(e), "Unexpected Error")
            self._usage_stats["agent_errors"].append((type(e).__name__, str(e)))

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
                self._usage_stats['ollama_response_error_count'] += 1
                self._interface.print_highlight(str(e), "Ollama ResponseError")
                if ollama_response_errors > MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS:
                    raise MaxOllamaRetrialsError(self.id.type, MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS)
                else:
                    self._interface.print_shadow("Ollama ResponseError -> retry (MAX_OLLAMA_RESPONSE_ERRORS_RETRIALS not yet reached)", "Ollama ResponseError")
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
                *[self._environment.execute_tool_call(self._tools, call, ctx.cancellation_token, self._interface, self._usage_stats, self.id.type)
                  for call in llm_res.content]
            )
            self._chat_history.append(FunctionExecutionResultMessage(content=tool_results))

        raise MaxIterationsError(self.id.type, self._max_internal_iterations)
