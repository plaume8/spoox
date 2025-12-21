from anthropic import BaseModel
from autogen_core.models import UserMessage


class PublicMessage(BaseModel):
    """Message containing the user prompt and triggering the singleton agent."""
    body: UserMessage
