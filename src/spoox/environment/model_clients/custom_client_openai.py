from typing import Sequence, Optional

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import LLMMessage, CreateResult, RequestUsage
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
        return RequestUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens
        )

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
                    "role": "user",
                    "content": m.content,
                })
            elif m.type == 'SystemMessage':
                parsed_messages.append({
                    "role": "developer",
                    "content": m.content,
                })
            elif m.type == 'AssistantMessageOpenAI':
                parsed_messages.extend(m.response_items)
            elif m.type == 'AssistantMessage':
                if isinstance(m.thought, str) and m.thought:
                    parsed_messages.append({
                        "role": "assistant",
                        "content": m.thought
                    })
                if isinstance(m.content, str):
                    parsed_messages.append({
                        "role": "assistant",
                        "content": m.content
                    })
                else:  # function calls
                    for function_call in m.content:
                        parsed_messages.append({
                            "type": "function_call",
                            "arguments": function_call.arguments,
                            "call_id": function_call.id,
                            "name": function_call.name,
                        })
            elif m.type == 'FunctionExecutionResultMessage':
                for f in m.content:
                    parsed_messages.append({
                        "type": "function_call_output",
                        "call_id": f.call_id,
                        "output": f.content,
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
            reasoning={"effort": "xhigh"},
            store=True,
            max_output_tokens=100000,
        )

        # parse token usage
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage is not None:
            prompt_tokens += response.usage.input_tokens
            completion_tokens += response.usage.output_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

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
                thought='; \n\n'.join(output_texts + thoughts),
                response_items=response.output
            )
        return CreateResultOpenAI(
            finish_reason="stop",
            content='; \n\n'.join(output_texts),
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            cached=False,
            thought='; \n\n'.join(thoughts),
            response_items=response.output
        )
