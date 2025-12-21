import asyncio
import uuid
from pathlib import Path

from autogen_core import DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core.models import UserMessage, ChatCompletionClient

from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak
from spoox.agents.mas.StructuredFlow.agents.SolverAgent import SolverAgent
from spoox.agents.mas.StructuredFlow.agents.SummarizerAgent import SummarizerAgent
from spoox.agents.mas.StructuredFlow.agents.TesterAgent import TesterAgent
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class SpooxSmall(AgentSystem):
    """
    This agent system implementation is based on the spoox-s multi-agent architecture,
    described in the spoox scaling study.
    It consists of three agents and a single feedback loop.
    """

    # all topic types
    group_chat_topic_type = "groupchat"
    solver_topic_type = "solver"
    tester_topic_type = "tester"
    summarizer_topic_type = "summarizer"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):
        super().__init__(interface, model_client, environment, timeout, logs_dir)
        # agents
        self._solver_agent = None
        self._tester_agent = None
        self._summarizer_agent = None

    async def _build_agents(self):
        """Initializing all agents, including all message subscriptions."""

        self._solver_agent = await SolverAgent.register(
            self.runtime,
            self.solver_topic_type,
            lambda: SolverAgent(
                topic_type=self.solver_topic_type,
                group_chat_topic_type=self.group_chat_topic_type,
                environment=self.environment,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                tester_agent_topic_type=self.tester_topic_type,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.solver_topic_type, agent_type=self._solver_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._solver_agent.type))

        self._tester_agent = await TesterAgent.register(
            self.runtime,
            self.tester_topic_type,
            lambda: TesterAgent(
                topic_type=self.tester_topic_type,
                group_chat_topic_type=self.group_chat_topic_type,
                environment=self.environment,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                previous_agent_topic_type=self.solver_topic_type,
                next_agent_topic_type=self.summarizer_topic_type,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.tester_topic_type, agent_type=self._tester_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._tester_agent.type))

        self._summarizer_agent = await SummarizerAgent.register(
            self.runtime,
            self.summarizer_topic_type,
            lambda: SummarizerAgent(
                topic_type=self.summarizer_topic_type,
                group_chat_topic_type=self.group_chat_topic_type,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.summarizer_topic_type, agent_type=self._summarizer_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._summarizer_agent.type))

    async def _trigger_agents(self, user_input: str) -> None:
        """Triggers the execution flow of the agent system's single agents, given the latest user input."""

        await self.runtime.publish_message(
            message=GroupChatMessage(nonce=str(uuid.uuid4()), body=UserMessage(content=user_input, source="User")),
            topic_id=DefaultTopicId(type=self.group_chat_topic_type)
        )
        # 0.1 delay to ensure the GroupChatMessage can be observed before the RequestToSpeak
        # (I think it is not required, however, it certainly does not hurt)
        await asyncio.sleep(0.1)
        await self.runtime.publish_message(
            message=RequestToSpeak(nonce=str(uuid.uuid4())),
            topic_id=DefaultTopicId(type=self.explorer_topic_type)
        )

    def get_state(self):
        """Returns the current state of the agent system for logging and later analysis."""
        return {
            'solver_agent': self._solver_agent,
            'tester_agent': self._tester_agent,
            'summarizer_agent': self._summarizer_agent,
        }
