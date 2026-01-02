import os
from pathlib import Path
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient

from spoox.agents import AgentSystem
from spoox.agents import SpooxLarge
from spoox.agents import SpooxMedium
from spoox.agents import SpooxSmall
from spoox.agents import SingletonAgentSystem
from spoox.environment import Environment
from spoox.interface import Interface


def setup_model_client(client_id: str, model_id: str) -> ChatCompletionClient:
    """
    Based on the provided client_id and model_id, the corresponding model client instance is created.

    :param client_id: The base model client, options: 'ollama', 'openai', 'anthropic'.
    :param model_id: the actual model id (e.g. 'qwen3:8b', 'claude-sonnet-4-5-20250929').
    :return: Model client ready to be used by the agent system.
    """

    if client_id == 'ollama':
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

    if client_id == 'openai':
        return OpenAIChatCompletionClient(model=model_id)

    if client_id == 'anthropic':
        return AnthropicChatCompletionClient(model=model_id)

    raise ValueError(f"No model client could be set up for: '{client_id}', '{model_id}'.")


def setup_agent_system(agent_system_id: str, model_client: ChatCompletionClient,
                       environment: Environment, interface: Interface,
                       timeout: int = 600, logs_dir: Path = Path.cwd()) -> AgentSystem:
    """Based on the provided 'agent_id', create the corresponding agent system instance."""

    if agent_system_id == "singleton":
        return SingletonAgentSystem(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-s":
        return SpooxSmall(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-m":
        return SpooxMedium(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-l":
        return SpooxLarge(interface, model_client, environment, timeout, logs_dir)
    raise ValueError(f"Selected agent system '{agent_system_id}' not known.")
