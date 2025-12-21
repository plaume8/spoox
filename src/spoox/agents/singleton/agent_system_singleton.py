import asyncio
import time
from pathlib import Path

from autogen_core import SingleThreadedAgentRuntime, DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core.models import UserMessage, ChatCompletionClient

from spoox.agents.agent_system import AgentSystem
from spoox.agents.singleton.messages import PublicMessage
from spoox.agents.singleton.singelton_agent import SingletonAgent
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class SingletonAgentSystem(AgentSystem):
    """
    This is the simplest agent system, consisting of a single agent.
    After the user submits a prompt, the singleton agent is executed, performs its work
    across multiple internal iterations, and terminates upon completion.
    Once the process is finished, the user may submit a new follow-up prompt.
    """

    singleton_topic_type = "singleton"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):

        super().__init__(interface, model_client, environment, timeout, logs_dir)
        self.runtime = SingleThreadedAgentRuntime()
        self._singleton_agent = None

    async def _build_agent(self):
        """Initializing all agents, including all message subscriptions."""

        self._singleton_agent = await SingletonAgent.register(
            self.runtime,
            self.singleton_topic_type,
            lambda: SingletonAgent(
                environment=self.environment,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.singleton_topic_type, agent_type=self._singleton_agent.type))

    async def start(self):
        """
        Run the agent system.
        The agent system is initialized and enters an infinite loop that alternates between waiting for user input
        and executing the singleton agent. The system exits when the user enters 'q', 'exit', or 'stop'.
        Furthermore, the system ensures that if a configured timeout is exceeded,
        the agent is notified and requested to stop as soon as possible, by setting the `_timeout_event`.
        """

        # agent system setup
        await self.environment.start()
        await self._build_agent()
        self.save_logs()
        start_time = time.time()

        # user input and agent call loop
        while True:

            # request user intput and trigger singleton agent
            user_input = self.interface.request_user_input("Query...")
            if user_input in ['q', 'exit', 'stop']:
                break
            self.runtime.start()
            await self.runtime.publish_message(
                message=PublicMessage(body=UserMessage(content=user_input, source="User")),
                topic_id=DefaultTopicId(type=self.singleton_topic_type),
            )

            self._start_timeout_countdown()
            await self.runtime.stop_when_idle()
            self._cancel_timeout_countdown()
            self.save_logs()

        # stop entirely
        await self.environment.stop()
        await self.runtime.close()
        # final logs
        self.save_logs(stopped=True, exec_time_sec=time.time() - start_time)

    def get_state(self):
        """Returns the current state of the agent system for logging and later analysis."""
        return {
            'single_agent_type': self._singleton_agent
        }
