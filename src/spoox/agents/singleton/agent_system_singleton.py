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
    After the user submits a prompt, the agent is executed, performs its work across multiple internal iterations,
    and terminates upon completion. Once the process is finished, the user may submit a new follow-up prompt.
    """

    singleton_topic_type = "singleton"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 3600, logs_dir: Path = Path.cwd()):

        super().__init__(interface, model_client, environment, timeout, logs_dir)
        self.runtime = SingleThreadedAgentRuntime()
        # timeout event that signals agents to return next time possible
        self._timeout_event = None
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
        if self._timeout_event is None:
            self._timeout_event = asyncio.Event()
        await self.environment.start()
        await self._build_agent()
        self.save_logs()
        start_time = time.time()

        # user input and agent call loop
        while True:

            # reset timeout event
            if self._timeout_event.is_set():
                self._timeout_event.clear()

            # request user intput and trigger singleton agent
            user_input = self.interface.request_user_input("Query...")
            if user_input in ['q', 'exit', 'stop']:
                break
            self.runtime.start()
            await self.runtime.publish_message(
                message=PublicMessage(body=UserMessage(content=user_input, source="User")),
                topic_id=DefaultTopicId(type=self.singleton_topic_type),
            )

            # wait until the agents are complete (runtime is idle);
            # if we just stop the runtime, but agents are still working on it, autogen runtime will raise a ValueError;
            # therefore, we use an event to signal all agents to stop the next time possible
            async def _timeout():
                await asyncio.sleep(self.timeout)
                error_message = "Agent System waiting for runtime.stop_when_idle timeout error"
                self.interface.print_highlight(error_message, "TimeoutError")
                self.usage_stats["agent_errors"].append(("TimeoutError", error_message))
                self._timeout_event.set()
            timeout_task = asyncio.create_task(_timeout())
            await self.runtime.stop_when_idle()
            timeout_task.cancel()
            self.save_logs()

        # stop entirely
        await self.environment.stop()
        await self.runtime.close()
        # final logs
        self.save_logs(stopped=True, exec_time_sec=time.time() - start_time)

    def init_usage_stats(self):
        self.usage_stats['llm_calls_count'] = 0
        self.usage_stats['tool_call_counts'] = dict()
        self.usage_stats['tool_calls'] = []
        self.usage_stats['ollama_response_error_count'] = 0
        self.usage_stats['model_client_exceptions'] = []
        self.usage_stats['agent_errors'] = []
        self.usage_stats['prompt_tokens'] = []
        self.usage_stats['completion_tokens'] = []

    def get_state(self):
        return {
            'single_agent_type': self._singleton_agent
        }
