import json
import pickle
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

from autogen_core.models import ChatCompletionClient
from spoox.environment.Environment import Environment
from spoox.interface.Interface import Interface


class AgentSystem(ABC):

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 3600, logs_dir: Path = Path.cwd()):
        self.interface = interface
        self.model_client = model_client
        self.environment = environment
        self.timeout = timeout

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.logs_dir = logs_dir / f"spoox_logs_{timestamp}"
        self.logs_dir.mkdir(parents=True)

        self.usage_stats = dict()
        self.init_usage_stats()

    @abstractmethod
    async def start(self) -> None:
        """Run the agent system."""
        pass

    @abstractmethod
    def init_usage_stats(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> dict:
        pass

    def save_logs(self, stopped: bool = False, exec_time_sec: int = 0) -> None:
        """
        Store agent execution logs.
        Execution time is under 1 ms, making it suitable for frequent use during agent operation.
        """

        with (self.logs_dir / f"meta_data.json").open("w") as f:
            meta_data = {
                "agent-system-type": self.environment.__class__.__name__,
                "model-info": self.model_client.model_info,
                "environment-type": self.environment.__class__.__name__,
                "interface-type": self.interface.__class__.__name__,
                "timeout": self.timeout,
                "model-client-actual-usage-prompt-tokens": self.model_client.actual_usage().prompt_tokens,
                "model-client-actual-usage-completion-tokens": self.model_client.actual_usage().completion_tokens,
                "model-client-total-usage-prompt-tokens": self.model_client.total_usage().prompt_tokens,
                "model-client-total-usage-completion-tokens": self.model_client.total_usage().completion_tokens,
                "agent-system-stopped": stopped,
                "agent-system-exec-time-sec": exec_time_sec,
            }
            json.dump(meta_data, f, indent=4)
        # save the agent_usage_stats dict in a pickle file
        with (self.logs_dir / f"usage_stats.pkl").open("wb") as f:
            pickle.dump(self.usage_stats, f)
        # save the agent system state as a dict in a pickle file
        with (self.logs_dir / f"agent_state.pkl").open("wb") as f:
            pickle.dump(self.get_state(), f)
        # save the interface logs in a pickle file
        with (self.logs_dir / f"interface.pkl").open("wb") as f:
            pickle.dump(self.interface, f)
