from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_SOLVER_SYSTEM_MESSAGE


class SolverAgent(BaseGroupChatAgent):
    """
    As its name implies, this agent is responsible for actively solving the task.
    It continues working until the task is completed or no further meaningful contributions are possible.
    The agent is equipped with all available tools to ensure the maximum possible action space.
    """

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
