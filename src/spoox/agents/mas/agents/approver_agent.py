from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.agents.prompts import get_APPROVER_SYSTEM_MESSAGE
from spoox.agents.mas.base_agent import BaseGroupChatAgent


class ApproverAgent(BaseGroupChatAgent):
    """
    The Approver's role is to examine the implemented solution in detail
    and decide whether the overall task has been completed entirely.
    He has no access to previous agent summaries, only the user's initial task description is provided.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            solver_agent_topic_type: str,
            next_agent_topic_type: str,
    ) -> None:

        next_agent_topic_types = [solver_agent_topic_type, next_agent_topic_type]

        system_message = get_APPROVER_SYSTEM_MESSAGE(
            topic_type, solver_agent_topic_type, next_agent_topic_type,
            agent_system.environment.get_additional_tool_descriptions(self))

        super().__init__(
            description="Agent tasked to decide if the implementation is complete and correct.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=next_agent_topic_types,
            max_internal_iterations=200,
            only_track_user_messages=True
        )
