from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_SUB_TASK_PLANNER_SYSTEM_MESSAGE


class SubTaskPlannerAgent(BaseGroupChatAgent):
    """
    Motivated by the idea of incorporating task decomposition into the MAS,
    the Sub-Task Planner identifies and formulates the next appropriate sub-task given the system’s current progress.
    Its role is to describe the sub-task and sketch a high-level approach for solving it,
    expressed purely in natural language, without offering a detailed step-by-step plan.
    To reinforce this abstraction-focused behavior, the agent is intentionally not equipped with any tools.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            explorer_topic_type: str,
            solver_topic_type: str,
            tester_topic_type: str,
    ) -> None:

        system_message = get_SUB_TASK_PLANNER_SYSTEM_MESSAGE(
            topic_type, explorer_topic_type, solver_topic_type, tester_topic_type)

        super().__init__(
            description="Agent tasked to create a plan for solving the task or a sub-task.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[explorer_topic_type, solver_topic_type, tester_topic_type],
            max_internal_iterations=10,
        )
