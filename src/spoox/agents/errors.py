class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, agent_id: str, message: str):
        m = f"Agent exception - for agent: {agent_id} - {message}"
        super().__init__(m)
        self.agent_id = agent_id


class ModelClientError(AgentError):

    def __init__(self, agent_id: str, max_retrials: int, model_client_exception: Exception):
        super().__init__(agent_id,
                         f"Model client exception:\n{model_client_exception} and max reached ({max_retrials})")
        self.max_retrials = max_retrials
        self.model_client_exception = model_client_exception


class MaxOnlyTextMessagesError(AgentError):

    def __init__(self, agent_id: str, max_only_text_messages: int):
        super().__init__(
            agent_id, f"Max only text messages reached ({max_only_text_messages})")
        self.max_only_text_messages = max_only_text_messages


class MaxIterationsError(AgentError):

    def __init__(self, agent_id: str, max_iterations: int):
        super().__init__(agent_id, f"Max agent iterations reached ({max_iterations})")
        self.max_iterations = max_iterations


class MaxOllamaRetrialsError(AgentError):

    def __init__(self, agent_id: str, max_ollama_retrials: int):
        super().__init__(agent_id,
                         f"Max ollama response errors retrials reached ({max_ollama_retrials})")
        self.max_ollama_retrials = max_ollama_retrials
