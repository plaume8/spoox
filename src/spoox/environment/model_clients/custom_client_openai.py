from typing import Sequence, Optional

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import LLMMessage, RequestUsage
from autogen_core.tools import Tool, ToolSchema
from openai import OpenAI
from openai.types.responses import FunctionToolParam

from spoox.environment.model_clients.custom_response_types import CreateResultOpenAI


class CustomOpenAIResponseAPIClient:
    # todo cleanup

    def __init__(self, model_id: str):

        self._client = OpenAI()
        self._model_id = model_id
        self.model_info = {'model_id': model_id}
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    async def create(
            self,
            messages: Sequence[LLMMessage],
            tools: Sequence[Tool | ToolSchema] = [],
            cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResultOpenAI:

        # parse messages
        parsed_messages = list()
        for m in messages:
            if m.type == 'UserMessage':
                parsed_messages.append({
                    "type": "message",
                    "role": "user",
                    "content": m.content,
                })
            elif m.type == 'SystemMessage':
                parsed_messages.append({
                    "type": "message",
                    "role": "developer",
                    "content": m.content,
                })
            elif m.type == 'AssistantMessageOpenAI':
                parsed_messages.extend(m.response_items)
            elif m.type == 'AssistantMessage' and isinstance(m.content, str):
                # if it is an AssistantMessage and no AssistantMessageOpenAI -> it always only contains
                # the summary of the previous agent
                parsed_messages.append({
                    "type": "message",
                    "role": "assistant",
                    "content": m.content,
                    "phase": "commentary"
                })
            elif m.type == 'FunctionExecutionResultMessage':
                for fR in m.content:
                    parsed_messages.append({
                        "type": "function_call_output",
                        "call_id": fR.call_id,
                        "output": fR.content,
                    })
            else:
                raise ValueError(f"Unexpected message type: {m.type}")

        # parse tools
        parsed_tools = list()
        for t in tools:
            t_schema = t.schema
            tool: FunctionToolParam = {
                "type": "function",
                "name": t_schema["name"],
                "description": t_schema["description"],
                "parameters": t_schema["parameters"],
                "strict": t_schema.get("strict", True),
            }
            parsed_tools.append(tool)

        # request model
        response = self._client.responses.create(
            model=self._model_id,
            input=parsed_messages,
            tools=parsed_tools,
            reasoning={"effort": "high"},
            store=True,
            max_output_tokens=40000
        )

        # parse token usage
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage is not None:
            prompt_tokens += response.usage.input_tokens
            completion_tokens += response.usage.output_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            print("Ran out of tokens")  # todo check and delete


        # parse response
        func_calls: list[FunctionCall] = list()
        output_texts: list[str] = list()
        thoughts: list[str] = list()
        for item in response.output:
            if item.type == "message":
                output_texts.extend(i.text for i in item.content or [] if i.type == "output_text")
            elif item.type == "reasoning":
                thoughts.extend(i.text for i in item.content or [])
            elif item.type == "function_call":
                func_calls.append(FunctionCall(
                    id=item.call_id,
                    arguments=item.arguments,
                    name=item.name,
                ))
            else:
                raise ValueError(f"Unknown response.output type {item.type} - {item}")

        if len(func_calls) > 0:
            return CreateResultOpenAI(
                finish_reason="function_calls",
                content=func_calls,
                usage=RequestUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
                cached=False,
                thought=';\n'.join(output_texts + thoughts),
                response_items=response.output
            )
        return CreateResultOpenAI(
            finish_reason="stop",
            content=';\n'.join(output_texts),
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            cached=False,
            thought=';\n'.join(thoughts),
            response_items=response.output
        )
