from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.agents.prompts import get_APPROVER_SYSTEM_MESSAGE
from spoox.agents.mas.base_agent import BaseGroupChatAgent


class ApproverAgent(BaseGroupChatAgent):
    """
    The Approver's role is to examine the entire solution progress and objectively decide
    whether the overall task has been completed. It is not equipped with any tools.
    It functions as a quality guard that does not actively interact with the environment.
    Instead, it exclusively focuses on the overall progress based on the entire group chat history.
    """

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
