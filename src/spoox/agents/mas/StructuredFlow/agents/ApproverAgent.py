from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.prompts import get_APPROVER_SYSTEM_MESSAGE


class ApproverAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            solver_agent_topic_type: str,
            test_agent_topic_type: str,
            next_agent_topic_type: str,
    ) -> None:

        next_agent_topic_types = [test_agent_topic_type, next_agent_topic_type]
        if solver_agent_topic_type:
            next_agent_topic_types.append(solver_agent_topic_type)

        system_message = get_APPROVER_SYSTEM_MESSAGE(
            topic_type, solver_agent_topic_type, test_agent_topic_type, next_agent_topic_type)

        super().__init__(
            description="Agent tasked to decide if the agents have done enough work on the task.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=next_agent_topic_types,
            max_internal_iterations=10,
        )
