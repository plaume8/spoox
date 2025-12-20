import asyncio
from typing import Callable

from autogen_core.models import ChatCompletionClient

from spoox.agents.mas.BaseGroupChatAgent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.utils import get_EXPLORER_SYSTEM_MESSAGE
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class ExplorerAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            group_chat_topic_type: str,
            environment: Environment,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            next_agent_topic: str,
            return_next_time_possible_event: asyncio.Event,
            support_feedback: bool = False,
    ) -> None:

        super().__init__(
            group_chat_topic_type=group_chat_topic_type,
            description="Agent tasked with exploring the system via its tools.",
            system_message=get_EXPLORER_SYSTEM_MESSAGE(topic_type, next_agent_topic,
                                                       environment.get_additional_tool_descriptions(self),
                                                       support_feedback),
            environment=environment,
            model_client=model_client,
            interface=interface,
            usage_stats=usage_stats,
            save_logs_f=save_logs_f,
            return_next_time_possible_event=return_next_time_possible_event,
            next_agent_topic_types=[next_agent_topic],
            max_internal_iterations=100,
        )
