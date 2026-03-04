import os
from enum import Enum
from pathlib import Path

from autogen_core.models import ChatCompletionClient

from spoox.agents import AgentSystem
from spoox.agents import SingletonAgentSystem
from spoox.agents import SpooxLarge
from spoox.agents import SpooxMedium
from spoox.agents import SpooxSmall
from spoox.environment import Environment
from spoox.environment.model_clients.custom_clients import CustomOpenAIResponseAPIClient
from spoox.interface import Interface


class AgentSystemId(Enum):
    """All available agent system ids."""

    SINGLETON = 'singleton'
    SPOOX_S = 'spoox-s'
    SPOOX_M = 'spoox-m'
    SPOOX_L = 'spoox-l'


def setup_agent_system(agent_system_id: AgentSystemId,
                       model_client: ChatCompletionClient | CustomOpenAIResponseAPIClient,
                       environment: Environment,
                       interface: Interface,
                       timeout: int = 600,
                       logs_dir: Path = Path.cwd()) -> AgentSystem:
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
