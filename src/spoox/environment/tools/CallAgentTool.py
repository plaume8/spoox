import asyncio
import uuid
from typing import Optional

from autogen_core import CancellationToken, Component, DefaultTopicId
from autogen_core.models import UserMessage
from autogen_core.tools import BaseTool
from proxytypes import func
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self

from spoox.agents.mas.Message import GroupChatMessage, RequestToSpeak

_PUBLISH_MESSAGE_FUNCS: dict[str, func] = {}


def set_publish_message_func(sender_agent_id: str, publish_message: func) -> None:
    global _PUBLISH_MESSAGE_FUNCS
    _PUBLISH_MESSAGE_FUNCS[sender_agent_id] = publish_message


def get_publish_message_funct(sender_agent_id: str) -> func:
    if sender_agent_id not in _PUBLISH_MESSAGE_FUNCS:
        raise ValueError(f"CallAgentTool: no publish message func set for agent {sender_agent_id}.")
    return _PUBLISH_MESSAGE_FUNCS[sender_agent_id]


class CallAgentInput(BaseModel):
    agent_tag: str = Field(description="The name of the agent that needs to be called.")
    sub_task: str = Field(description="The sub-task the agent is expected to work on, "
                                      "outlining what the agent should do and achieve.")


class CallAgentResult(BaseModel):
    success: bool
    called_agent_id: str
    error_message: Optional[str] = None

    @model_serializer
    def ser_model(self) -> str:
        if self.success:
            return f"Agent {self.called_agent_id.capitalize()} successfully called."
        return f"Failed to call {self.called_agent_id.capitalize()} agent: {self.error_message or ''}"


class CallAgentConfig(BaseModel):
    """Configuration for CallAgentTool"""

    autogen_runtime_key: str
    description: str = ("Tool for calling a specialized agent to work on a sub-task. "
                        "The agent requires a detailed description of the sub-task.")


class CallAgentTool(
    BaseTool[CallAgentInput, CallAgentResult],
    Component[CallAgentConfig]
):
    """
    A tool with which an agent can call another agent to work on an assigned sub-task.
    """

    component_config_schema = CallAgentConfig

    def __init__(self, known_agent_tags: list[str], sender_agent_id: str, group_chat_topic_type: str) -> None:
        super().__init__(
            CallAgentInput,
            CallAgentResult,
            "CallAgent",
            ("Tool for calling a specialized agent to work on a sub-task. "
             "The agent requires a detailed description of the sub-task.")
        )
        self._known_agent_tags = [t.lower() for t in known_agent_tags]
        self._sender_agent_id = sender_agent_id.lower()
        self._group_chat_topic_type = group_chat_topic_type
        self._publish_message: func = get_publish_message_funct(sender_agent_id)

    async def run(self, args: CallAgentInput, cancellation_token: CancellationToken = None) -> CallAgentResult:

        agent_tag = args.agent_tag.lower()
        sub_task = args.sub_task

        # check for agent tag (we check this way to be more resilient -> if agent does something wrong)
        matching_tags = [t for t in self._known_agent_tags if t in agent_tag]
        if len(matching_tags) == 0:
            return CallAgentResult(success=False, called_agent_id=agent_tag,
                                   error_message=f"Agent '{agent_tag}' not known.")
        if len(matching_tags) > 1:
            return CallAgentResult(success=False, called_agent_id=agent_tag,
                                   error_message=f"Multiple agents found, please call only one at a time: '{matching_tags}'.")
        agent_tag = matching_tags[0]

        # send GroupChat "info" message as well as RequestToSpeak message to agent
        info_message = f"{self._sender_agent_id.capitalize()} agent calls {agent_tag.capitalize()} agent to do the sub-task:\n {sub_task}"
        await self._publish_message(
            message=GroupChatMessage(nonce=str(uuid.uuid4()),
                                     body=UserMessage(content=info_message, source=self._sender_agent_id)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )
        await asyncio.sleep(0.1)  # ensuring the group msg can be observed before the RTS
        await self._publish_message(RequestToSpeak(nonce=str(uuid.uuid4())), DefaultTopicId(type=agent_tag))

        return CallAgentResult(success=True, called_agent_id=agent_tag)

    def _to_config(self) -> CallAgentConfig:
        """Convert current instance to config object"""
        return CallAgentConfig(known_agent_tags=self._known_agent_tags, sender_agent_id=self._sender_agent_id, group_chat_topic_type=self._group_chat_topic_type)

    @classmethod
    def _from_config(cls, config: CallAgentConfig) -> Self:
        """Create instance from config object"""
        return cls(known_agent_tags=config.known_agent_tags, sender_agent_id=config.sender_agent_id, group_chat_topic_type=config.group_chat_topic_type)
