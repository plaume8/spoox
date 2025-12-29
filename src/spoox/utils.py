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
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


def setup_model_client(model_id: str, docker_access: bool = False) -> ChatCompletionClient:
    """
    Based on the provided 'model_id', create the corresponding model client instance.
    Field `docker_access` should be set to True if Ollama is called from the inside of a docker container.
    """

    if docker_access:
        host = "http://host.docker.internal:11434"
    else:
        host = "http://localhost:11434"

    if model_id in ["gpt-oss:20b", "gpt-oss:120b", "magistral:24b"]:
        model_info = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
            "multiple_system_messages": False
        }
        return OllamaChatCompletionClient(model=model_id, model_info=model_info, host=host)

    if model_id in ["qwen3:8b", "qwen3:14b", "mistral-nemo:12b"]:
        return OllamaChatCompletionClient(model=model_id, host=host)
    if model_id == "claude-sonnet-4":
        return AnthropicChatCompletionClient(model="claude-sonnet-4-20250514")
    if model_id == "claude-sonnet-4-5":
        return AnthropicChatCompletionClient(model="claude-sonnet-4-5-20250929")
    if model_id.startswith('gpt-5'):
        return OpenAIChatCompletionClient(model=model_id)

    raise ValueError(f"Selected model '{model_id}' not known.")


def setup_agent_system(agent_id: str, model_client: ChatCompletionClient,
                       environment: Environment, interface: Interface,
                       timeout: int = 600, logs_dir: Path = Path.cwd()) -> AgentSystem:
    """Based on the provided 'agent_id', create the corresponding agent system instance."""

    if agent_id == "singleton":
        return SingletonAgentSystem(interface, model_client, environment, timeout, logs_dir)
    if agent_id == "mas-group-chat-s":
        return SpooxSmall(interface, model_client, environment, timeout, logs_dir)
    if agent_id == "mas-group-chat-m":
        return SpooxMedium(interface, model_client, environment, timeout, logs_dir)
    if agent_id == "mas-group-chat-l":
        return SpooxLarge(interface, model_client, environment, timeout, logs_dir)
    raise ValueError(f"Selected agent '{agent_id}' not known.")

