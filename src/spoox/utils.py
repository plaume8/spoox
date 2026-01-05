import os
from enum import Enum
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


class ModelClientId(Enum):
    """All available model client ids."""

    ANTHROPIC = 'anthropic'
    OLLAMA = 'ollama'
    OPENAI = 'openai'


class AgentSystemId(Enum):
    """All available agent system ids."""

    SINGLETON = 'singleton'
    SPOOX_S = 'spoox-s'
    SPOOX_M = 'spoox-m'
    SPOOX_L = 'spoox-l'



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
        return OpenAIChatCompletionClient(model=model_id)

    if client_id == ModelClientId.ANTHROPIC:
        return AnthropicChatCompletionClient(model=model_id)

    raise ValueError(f"No model client could be set up for: '{client_id}', '{model_id}'.")


def _check_env(client_id: ModelClientId) -> None:
    """Check if the environment is set up correctly for given model client id."""
    if client_id == ModelClientId.OLLAMA and "OLLAMA" not in os.environ:
        raise ValueError(f"Required environment variable 'OLLAMA' is not set.")
    elif client_id == ModelClientId.OPENAI and "OPENAI_API_KEY" not in os.environ:
        raise ValueError(f"Required environment variable 'OPENAI_API_KEY' is not set.")
    elif client_id == ModelClientId.ANTHROPIC and "'ANTHROPIC_API_KEY'" not in os.environ:
        raise ValueError(f"Required environment variable 'ANTHROPIC_API_KEY' is not set.")


def setup_agent_system(agent_system_id: AgentSystemId, model_client: ChatCompletionClient,
                       environment: Environment, interface: Interface,
                       timeout: int = 600, logs_dir: Path = Path.cwd()) -> AgentSystem:
    """Based on the provided 'agent_id', create the corresponding agent system instance."""

    if agent_system_id == AgentSystemId.SINGLETON:
        return SingletonAgentSystem(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == AgentSystemId.SPOOX_S:
        return SpooxSmall(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == AgentSystemId.SPOOX_M:
        return SpooxMedium(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == AgentSystemId.SPOOX_L:
        return SpooxLarge(interface, model_client, environment, timeout, logs_dir)
    raise ValueError(f"Selected agent system '{agent_system_id}' not known.")
