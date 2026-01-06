import asyncio
import uuid
from pathlib import Path

from autogen_core import DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core.models import UserMessage, ChatCompletionClient

from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak, GROUP_CHAT_TOPIC_TYPE
from spoox.agents.mas.agents import ApproverAgent
from spoox.agents.mas.agents import ExplorerAgent
from spoox.agents.mas.agents import SubTaskSolverAgent
from spoox.agents.mas.agents import SubTaskPlannerAgent
from spoox.agents.mas.agents import SummarizerAgent
from spoox.agents.mas.agents import TesterAgent
from spoox.agents.mas.agents import RefinerAgent
from spoox.environment import Environment
from spoox.interface import Interface


class SpooxLarge(AgentSystem):
    """
    This agent system implementation is based on the spoox-l multi-agent architecture,
    described in the spoox scaling study.
    It consists of seven agents and five feedback loops.
    """

    # all topic types
    explorer_topic_type = "explorer"
    sub_task_planner_topic_type = "subtaskplanner"
    sub_task_solver_topic_type = "subtasksolver"
    tester_topic_type = "tester"
    refiner_topic_type = "refiner"
    approver_topic_type = "approver"
    summarizer_topic_type = "summarizer"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):
        super().__init__(interface, model_client, environment, timeout, logs_dir)
        # agents
        self._explorer_agent = None
        self._sub_task_planner_agent = None
        self._sub_task_solver_agent = None
        self._tester_agent = None
        self._refiner_agent = None
        self._approver_agent = None
        self._summarizer_agent = None

    async def _build_agents(self):
        """Initializing all agents, including all message subscriptions."""

        self._explorer_agent = await ExplorerAgent.register(
            self.runtime,
            self.explorer_topic_type,
            lambda: ExplorerAgent(
                topic_type=self.explorer_topic_type,
                agent_system=self,
                next_agent_topic=self.sub_task_planner_topic_type,
                support_feedback=True,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.explorer_topic_type, agent_type=self._explorer_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._explorer_agent.type))

        self._sub_task_planner_agent = await SubTaskPlannerAgent.register(
            self.runtime,
            self.sub_task_planner_topic_type,
            lambda: SubTaskPlannerAgent(
                topic_type=self.sub_task_planner_topic_type,
                agent_system=self,
                explorer_topic_type=self.explorer_topic_type,
                solver_topic_type=self.sub_task_solver_topic_type,
                tester_topic_type=self.tester_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.sub_task_planner_topic_type, agent_type=self._sub_task_planner_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._sub_task_planner_agent.type))

        self._sub_task_solver_agent = await SubTaskSolverAgent.register(
            self.runtime,
            self.sub_task_solver_topic_type,
            lambda: SubTaskSolverAgent(
                topic_type=self.sub_task_solver_topic_type,
                agent_system=self,
                planner_agent_topic_type=self.sub_task_planner_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.sub_task_solver_topic_type, agent_type=self._sub_task_solver_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._sub_task_solver_agent.type))

        self._tester_agent = await TesterAgent.register(
            self.runtime,
            self.tester_topic_type,
            lambda: TesterAgent(
                topic_type=self.tester_topic_type,
                agent_system=self,
                previous_agent_topic_type=self.refiner_topic_type,
                next_agent_topic_type=self.refiner_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.tester_topic_type, agent_type=self._tester_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._tester_agent.type))

        self._refiner_agent = await RefinerAgent.register(
            self.runtime,
            self.refiner_topic_type,
            lambda: RefinerAgent(
                topic_type=self.refiner_topic_type,
                agent_system=self,
                tester_topic_type=self.tester_topic_type,
                approver_topic_type=self.approver_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.refiner_topic_type, agent_type=self._refiner_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._refiner_agent.type))

        self._approver_agent = await ApproverAgent.register(
            self.runtime,
            self.approver_topic_type,
            lambda: ApproverAgent(
                topic_type=self.approver_topic_type,
                agent_system=self,
                solver_agent_topic_type=self.refiner_topic_type,
                test_agent_topic_type=self.tester_topic_type,
                next_agent_topic_type=self.summarizer_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.approver_topic_type, agent_type=self._approver_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._approver_agent.type))

        self._summarizer_agent = await SummarizerAgent.register(
            self.runtime,
            self.summarizer_topic_type,
            lambda: SummarizerAgent(
                topic_type=self.summarizer_topic_type,
                agent_system=self,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.summarizer_topic_type, agent_type=self._summarizer_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=GROUP_CHAT_TOPIC_TYPE, agent_type=self._summarizer_agent.type))

    async def _trigger_agents(self, user_input: str) -> None:
        """Triggers the execution flow of the agent system's single agents, given the latest user input."""

        await self.runtime.publish_message(
            message=GroupChatMessage(nonce=str(uuid.uuid4()), body=UserMessage(content=user_input, source="User")),
            topic_id=DefaultTopicId(type=GROUP_CHAT_TOPIC_TYPE)
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
            'explorer_agent': self._explorer_agent,
            'sub_task_planner_agent': self._sub_task_planner_agent,
            'sub_task_solver_agent': self._sub_task_solver_agent,
            'tester_agent': self._tester_agent,
            'refiner_agent': self._refiner_agent,
            'approver_agent': self._approver_agent,
            'summarizer_agent': self._summarizer_agent,
        }
