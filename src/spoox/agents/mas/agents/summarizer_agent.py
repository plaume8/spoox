from spoox.agents.agent_system import AgentSystem
from spoox.agents.mas.base_agent import BaseGroupChatAgent
from spoox.agents.mas.agents.prompts import get_SUMMARIZER_SYSTEM_MESSAGE


class SummarizerAgent(BaseGroupChatAgent):
    """
    This agent has been introduced mainly for usability purposes, since clear, user-facing explanations contribute
    to more trustworthy AI4SE systems. Its only role is to craft a concise and user-oriented final summary
    based on the group chat history, without using any tools.
    """

    def __init__(
            self,
            topic_type: str,
            agent_system: AgentSystem,
    ) -> None:

        super().__init__(
            description="Agent creating the final summary.",
            system_message=get_SUMMARIZER_SYSTEM_MESSAGE(topic_type),
            agent_system=agent_system,
            max_internal_iterations=10,
        )
