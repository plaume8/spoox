import asyncio
from typing import Callable

from autogen_core.models import ChatCompletionClient

from spoox.agents.mas.BaseGroupChatAgent import BaseGroupChatAgent
from spoox.agents.mas.StructuredFlow.agents.utils import get_SUMMARIZER_SYSTEM_MESSAGE
from spoox.interface.Interface import Interface


class SummarizerAgent(BaseGroupChatAgent):

    def __init__(
            self,
            topic_type: str,
            group_chat_topic_type: str,
            model_client: ChatCompletionClient,
            interface: Interface,
            usage_stats: dict,
            save_logs_f: Callable,
            return_next_time_possible_event: asyncio.Event
    ) -> None:

        super().__init__(
            group_chat_topic_type=group_chat_topic_type,
            description="Agent creating the final summary.",
            system_message=get_SUMMARIZER_SYSTEM_MESSAGE(topic_type),
            model_client=model_client,
            interface=interface,
            usage_stats=usage_stats,
            save_logs_f=save_logs_f,
            max_internal_iterations=10,
            return_next_time_possible_event=return_next_time_possible_event
        )
