import asyncio
from typing import Callable

from autogen_core.models import ChatCompletionClient

from spoox.agents.mas.BaseGroupChatAgent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.utils import get_SUB_TASK_PLANNER_SYSTEM_MESSAGE
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class SubTaskPlannerAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            group_chat_topic_type: str,
            environment: Environment,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            explorer_topic_type: str,
            solver_topic_type: str,
            tester_topic_type: str,
            return_next_time_possible_event: asyncio.Event
    ) -> None:

        super().__init__(
            group_chat_topic_type=group_chat_topic_type,
            description="Agent tasked to create a plan for solving the task or a sub-task.",
            system_message=get_SUB_TASK_PLANNER_SYSTEM_MESSAGE(topic_type, explorer_topic_type, solver_topic_type, tester_topic_type),
            environment=environment,
            model_client=model_client,
            interface=interface,
            usage_stats=usage_stats,
            save_logs_f=save_logs_f,
            return_next_time_possible_event=return_next_time_possible_event,
            next_agent_topic_types=[explorer_topic_type, solver_topic_type, tester_topic_type],
            max_internal_iterations=10,
        )
