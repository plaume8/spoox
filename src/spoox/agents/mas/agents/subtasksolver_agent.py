from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_SUB_TASK_SOLVER_SYSTEM_MESSAGE


class SubTaskSolverAgent(BaseGroupChatAgent):
    """
    Unlike the Solver, this agent should focus exclusively on completing the latest specified sub-task.
    It has access to the full toolset, and operates under the assumption that the sub-task has already been selected
    and sufficiently described by preceding agents.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
            planner_agent_topic_type: str,
    ) -> None:

        system_message = get_SUB_TASK_SOLVER_SYSTEM_MESSAGE(
            topic_type, planner_agent_topic_type, agent_system.environment.get_additional_tool_descriptions(self))

        super().__init__(
            description="Agent tasked to solve the given task or sub-task via its tools.",
            system_message=system_message,
            agent_system=agent_system,
            next_agent_topic_types=[planner_agent_topic_type],
            max_internal_iterations=100,
            reset_on_request_to_speak=True,
        )
