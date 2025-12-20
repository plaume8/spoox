from anthropic import BaseModel
from autogen_core.models import UserMessage


# all messages include 'nonce' to ensure each message is unique,
# preventing handlers from merging messages sent close together in time.


class GroupChatMessage(BaseModel):
    """Simple text message for a group chat. Typically distributed to all agents."""
    nonce: str
    body: UserMessage


class RequestToSpeak(BaseModel):
    """Simple organizational message requesting an agent to start working."""
    nonce: str
