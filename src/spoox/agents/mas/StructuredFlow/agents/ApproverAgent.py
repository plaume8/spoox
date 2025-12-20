import asyncio
from typing import Callable

from autogen_core.models import ChatCompletionClient

from spoox.agents.mas.BaseGroupChatAgent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.utils import get_APPROVER_SYSTEM_MESSAGE
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class ApproverAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            group_chat_topic_type: str,
            environment: Environment,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            return_next_time_possible_event: asyncio.Event,
            solver_agent_topic_type: str,
            test_agent_topic_type: str,
            next_agent_topic_type: str,
    ) -> None:

        next_agent_topic_types = [test_agent_topic_type, next_agent_topic_type]
        if solver_agent_topic_type:
            next_agent_topic_types.append(solver_agent_topic_type)

        super().__init__(
            group_chat_topic_type=group_chat_topic_type,
            description="Agent tasked to decide if the agents have done enough work on the task.",
            system_message=get_APPROVER_SYSTEM_MESSAGE(topic_type, solver_agent_topic_type,
                                                       test_agent_topic_type, next_agent_topic_type),
            environment=environment,
            model_client=model_client,
            interface=interface,
            usage_stats=usage_stats,
            save_logs_f=save_logs_f,
            return_next_time_possible_event=return_next_time_possible_event,
            next_agent_topic_types=next_agent_topic_types,
            max_internal_iterations=10,
        )
