from typing import List, Literal

from autogen_core.models import AssistantMessage, CreateResult
from openai.types.responses import ResponseOutputItem


class AssistantMessageOpenAI(AssistantMessage):
    """AutoGen AssistantMessage extended by openai response API context."""
    response_items: List[ResponseOutputItem]
    type: Literal["AssistantMessageOpenAI"] = "AssistantMessageOpenAI"


class CreateResultOpenAI(CreateResult):
    """AutoGen CreateResult extended by openai response API context."""
    response_items: List[ResponseOutputItem]
