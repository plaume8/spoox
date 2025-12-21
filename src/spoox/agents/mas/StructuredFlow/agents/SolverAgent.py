from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.prompts import get_SOLVER_SYSTEM_MESSAGE
from spoox.environment.Environment import Environment


class SolverAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            tester_agent_topic_type: str,
    ) -> None:

        system_message = get_SOLVER_SYSTEM_MESSAGE(
            topic_type, tester_agent_topic_type, agent_system.environment.get_additional_tool_descriptions(self))

        super().__init__(
            description="Agent tasked to solve the given task via its tools.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[tester_agent_topic_type],
            max_internal_iterations=100,
        )
