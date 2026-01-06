from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_EXPLORER_SYSTEM_MESSAGE


class ExplorerAgent(BaseGroupChatAgent):
    """
    This agent is prompted to solely focus on exploring the environment and gathering basic information about it.
    It is not intended to solve the task, its only objective is to produce a broad overview of the given environment.
    To emphasize its purely exploratory nature, this agent is intentionally not equipped with the Python tool.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            next_agent_topic: str,
            support_feedback: bool = False,
    ) -> None:

        system_message = get_EXPLORER_SYSTEM_MESSAGE(
            topic_type, next_agent_topic, agent_system.environment.get_additional_tool_descriptions(self), support_feedback)

        super().__init__(
            description="Agent tasked with exploring the system via its tools.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[next_agent_topic],
            max_internal_iterations=100,
        )
