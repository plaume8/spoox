from pathlib import Path

from autogen_core import DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core.models import UserMessage, ChatCompletionClient

from spoox.agents.agent_system import AgentSystem
from spoox.agents.singleton.messages import PublicMessage
from spoox.agents.singleton.singelton_agent import SingletonAgent
from spoox.environment import Environment
from spoox.interface import Interface


class SingletonAgentSystem(AgentSystem):
    """
    This is the simplest agent system, consisting of a single agent.
    After the user submits a prompt, the singleton agent is executed, performs its work
    across multiple internal iterations, and terminates upon completion.
    """

    singleton_topic_type = "singleton"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):

        super().__init__(interface, model_client, environment, timeout, logs_dir)
        self._singleton_agent = None

    async def _build_agents(self):
        """Initializing all agents, including all message subscriptions."""

        self._singleton_agent = await SingletonAgent.register(
            self.runtime,
            self.singleton_topic_type,
            lambda: SingletonAgent(self)
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.singleton_topic_type, agent_type=self._singleton_agent.type))

    async def _trigger_agents(self, user_input: str) -> None:
        """Triggers the execution flow of the agent system's single agents, given the latest user input."""

        await self.runtime.publish_message(
            message=PublicMessage(body=UserMessage(content=user_input, source="User")),
            topic_id=DefaultTopicId(type=self.singleton_topic_type),
        )

    def get_state(self):
        """Returns the current state of the agent system for logging and later analysis."""
        return {'single_agent_type': self._singleton_agent}
