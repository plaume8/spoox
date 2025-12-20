import asyncio
from typing import Callable

from autogen_core.models import ChatCompletionClient

from spoox.agents.mas.BaseGroupChatAgent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.utils import get_SOLVER_SYSTEM_MESSAGE
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class SolverAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            group_chat_topic_type: str,
            environment: Environment,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            tester_agent_topic_type: str,
            return_next_time_possible_event: asyncio.Event
    ) -> None:

        super().__init__(
            group_chat_topic_type=group_chat_topic_type,
            description="Agent tasked to solve the given task via its tools.",
            system_message=get_SOLVER_SYSTEM_MESSAGE(topic_type, tester_agent_topic_type,
                                                     environment.get_additional_tool_descriptions(self)),
            environment=environment,
            model_client=model_client,
            interface=interface,
            usage_stats=usage_stats,
            save_logs_f=save_logs_f,
            return_next_time_possible_event=return_next_time_possible_event,
            next_agent_topic_types=[tester_agent_topic_type],
            max_internal_iterations=100,
        )
