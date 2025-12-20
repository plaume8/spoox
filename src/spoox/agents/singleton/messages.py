from anthropic import BaseModel
from autogen_core.models import UserMessage


class PublicMessage(BaseModel):
    """Message sent by the user that triggers the singleton agent."""
    body: UserMessage
