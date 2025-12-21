import asyncio
import time
import uuid
from pathlib import Path

from autogen_core import SingleThreadedAgentRuntime, DefaultTopicId
from autogen_core import TypeSubscription
from autogen_core.models import UserMessage, ChatCompletionClient

from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.messages import GroupChatMessage, RequestToSpeak
from spoox.agents.mas.StructuredFlow.agents.ApproverAgent import ApproverAgent
from spoox.agents.mas.StructuredFlow.agents.ExplorerAgent import ExplorerAgent
from spoox.agents.mas.StructuredFlow.agents.SolverAgent import SolverAgent
from spoox.agents.mas.StructuredFlow.agents.SummarizerAgent import SummarizerAgent
from spoox.agents.mas.StructuredFlow.agents.TesterAgent import TesterAgent
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class UbuntuMASGroupChatMedium(AgentSystem):

    # all topic types
    group_chat_topic_type = "groupchat"
    explorer_topic_type = "explorer"
    solver_topic_type = "solver"
    tester_topic_type = "tester"
    approver_topic_type = "approver"
    summarizer_topic_type = "summarizer"

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):

        super().__init__(interface, model_client, environment, timeout, logs_dir)
        self.runtime = SingleThreadedAgentRuntime()
        # agents
        self._explorer_agent = None
        self._solver_agent = None
        self._tester_agent = None
        self._approver_agent = None
        self._summarizer_agent = None

    async def build_mas(self):
        """setup all agents"""

        self._explorer_agent = await ExplorerAgent.register(
            self.runtime,
            self.explorer_topic_type,
            lambda: ExplorerAgent(
                topic_type=self.explorer_topic_type,
                group_chat_topic_type=self.group_chat_topic_type,
                environment=self.environment,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                next_agent_topic=self.solver_topic_type,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.explorer_topic_type, agent_type=self._explorer_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._explorer_agent.type))

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
                next_agent_topic_type=self.approver_topic_type,
                return_next_time_possible_event=self._timeout_event,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.tester_topic_type, agent_type=self._tester_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._tester_agent.type))

        self._approver_agent = await ApproverAgent.register(
            self.runtime,
            self.approver_topic_type,
            lambda: ApproverAgent(
                topic_type=self.approver_topic_type,
                group_chat_topic_type=self.group_chat_topic_type,
                environment=self.environment,
                model_client=self.model_client,
                interface=self.interface,
                usage_stats=self.usage_stats,
                save_logs_f=self.save_logs,
                return_next_time_possible_event=self._timeout_event,
                solver_agent_topic_type=self.solver_topic_type,
                test_agent_topic_type=self.tester_topic_type,
                next_agent_topic_type=self.summarizer_topic_type,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.approver_topic_type, agent_type=self._approver_agent.type))
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.group_chat_topic_type, agent_type=self._approver_agent.type))

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

    async def start(self):

        await self.environment.start()
        await self.build_mas()
        self.save_logs()
        start_time = time.time()

        # user input loop
        while True:

            user_input = self.interface.request_user_input("Query...")
            if user_input in ['q', 'exit', 'stop']:
                break
            self.runtime.start()
            await self.runtime.publish_message(
                message=GroupChatMessage(nonce=str(uuid.uuid4()), body=UserMessage(content=user_input, source="User")),
                topic_id=DefaultTopicId(type=self.group_chat_topic_type)
            )
            # ensuring the group msg can be observed before the RTS (I think it is not required - but not sure...)
            await asyncio.sleep(0.1)
            await self.runtime.publish_message(
                message=RequestToSpeak(nonce=str(uuid.uuid4())),
                topic_id=DefaultTopicId(type=self.explorer_topic_type)
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
            'explorer_agent': self._explorer_agent,
            'solver_agent': self._solver_agent,
            'tester_agent': self._tester_agent,
            'approver_agent': self._approver_agent,
            'summarizer_agent': self._summarizer_agent,
        }
