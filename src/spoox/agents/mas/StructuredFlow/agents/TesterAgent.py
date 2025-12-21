from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.prompts import get_TESTER_SYSTEM_MESSAGE


class TesterAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            previous_agent_topic_type: str,
            next_agent_topic_type: str,
    ) -> None:

        system_message = get_TESTER_SYSTEM_MESSAGE(
            topic_type,
            previous_agent_topic_type,
            next_agent_topic_type,
            agent_system.environment.get_additional_tool_descriptions(self)
        )

        super().__init__(
            description="Agent tasked to test the solution of the given task.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[previous_agent_topic_type, next_agent_topic_type],
            max_internal_iterations=100,
        )
