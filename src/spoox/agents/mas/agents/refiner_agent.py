from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_REFINER_SYSTEM_MESSAGE


class RefinerAgent(BaseGroupChatAgent):
    """
    Equipped with all tools, this agent is responsible for resolving discovered bugs
    and refining or completing the existing implementation of the overall task.
    It assumes that previous agents have already identified bugs, missing components, and other errors in the solution.
    Its sole focus is to address and fix these most recent findings.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            tester_topic_type: str,
            approver_topic_type: str,
    ) -> None:

        system_message = get_REFINER_SYSTEM_MESSAGE(
            topic_type, tester_topic_type, approver_topic_type, agent_system.environment.get_additional_tool_descriptions(self))

        super().__init__(
            description="Agent tasked to refine and fix the implemented task solution.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[tester_topic_type, approver_topic_type],
            max_internal_iterations=100,
            reset_on_request_to_speak=True,
        )
