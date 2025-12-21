from anthropic import BaseModel
from autogen_core.models import UserMessage


# generic topic type used for group chat messages.
GROUP_CHAT_TOPIC_TYPE = "groupchat"


# all messages include 'nonce' to ensure each message is unique,
# preventing handlers from merging messages sent close together in time.

class GroupChatMessage(BaseModel):
    """Text message for a group chat. Typically distributed to all agents."""
    nonce: str
    body: UserMessage


class RequestToSpeak(BaseModel):
    """Organizational message requesting an agent to start working."""
    nonce: str
