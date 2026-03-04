import os
from enum import Enum
from typing import Sequence, Optional

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, ModelInfo, ModelFamily, LLMMessage, CreateResult
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from openai import OpenAI


class ModelClientId(Enum):
    """All available model client ids."""

    ANTHROPIC = 'anthropic'
    OLLAMA = 'ollama'
    OPENAI = 'openai'


def setup_model_client(client_id: ModelClientId, model_id: str) -> ChatCompletionClient:
    """
    Based on the provided client_id and model_id, the corresponding model client instance is created.

    Args:
        client_id (str): The base model client, options: 'ollama', 'openai', 'anthropic'.
        model_id (str): The actual model id (e.g. 'qwen3:8b', 'claude-sonnet-4-5-20250929').

    Returns:
        ChatCompletionClient: Model client ready to be used by the agent system.
    """

    _check_env(client_id)

    if client_id == ModelClientId.OLLAMA:
        # get ollama endpoint
        host = os.environ['OLLAMA']
        # special exception for gpt-oss models -> ollama not keeps a pre-set model_info -> todo test if still necessary
        if model_id in ["gpt-oss:20b", "gpt-oss:120b"]:
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
                "multiple_system_messages": False
            }
            return OllamaChatCompletionClient(model=model_id, model_info=model_info, host=host)
        return OllamaChatCompletionClient(model=model_id, host=host)

    if client_id == ModelClientId.OPENAI:
        if model_id == "gpt-5.3-codex":
            model_info: ModelInfo = {
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.GPT_5,
                "structured_output": True,
                "multiple_system_messages": True,
            }
            return OpenAIChatCompletionClient(model=model_id, model_info=model_info)
        return OpenAIChatCompletionClient(model=model_id)

    if client_id == ModelClientId.ANTHROPIC:
        return AnthropicChatCompletionClient(model=model_id)

    raise ValueError(f"No model client could be set up for: '{client_id}', '{model_id}'.")



class CustomOpenAIResponseAPIClient:

    def __init__(self, model_id: str):

        self._client = OpenAI()
        self._model_id = model_id

    def request(self, message: str):

        response = self._client.responses.create(
            model=self._model_id,
            input=message
        )
        print(response)

    def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        cancellation_token: Optional[CancellationToken] = None
    ) -> CreateResult:

        # parse messages
        history = list()
        for m in messages:
            if m.type == 'UserMessage':
                history.append({
                    "role": "user",
                    "content": m.content
                })
                # todo source arg
                assert (type(m.content) == str)
            if m.type == 'SystemMessage':
                history.append({
                    "role": "system",
                    "content": m.content
                })
            if m.type == 'AssistantMessage':
                history.append({
                    "role": "assistant",
                    "content": m.content
                })
                # todo source arg
                # todo function call
                # todo thought

            if m.type == 'FunctionExecutionResultMessage':
                # todo

        # request model
        history.append({})
        response = self._client.responses.create(
            model=self._model_id,
            input=history
        )

        # parse response


if __name__ == '__main__':
    model_client = CustomOpenAIResponseAPIClient(model_id="gpt-5-mini")

    model_client.request("What is your name")



def _check_env(client_id: ModelClientId) -> None:
    """Check if the environment is set up correctly for given model client id."""
    if client_id == ModelClientId.OLLAMA and "OLLAMA" not in os.environ:
        raise ValueError(f"Required environment variable 'OLLAMA' is not set.")
    elif client_id == ModelClientId.OPENAI and "OPENAI_API_KEY" not in os.environ:
        raise ValueError(f"Required environment variable 'OPENAI_API_KEY' is not set.")
    elif client_id == ModelClientId.ANTHROPIC and "'ANTHROPIC_API_KEY'" not in os.environ:
        raise ValueError(f"Required environment variable 'ANTHROPIC_API_KEY' is not set.")

