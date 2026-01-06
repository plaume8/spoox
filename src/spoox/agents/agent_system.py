import asyncio
import json
import pickle
import time
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

from autogen_core import SingleThreadedAgentRuntime
from autogen_core.models import ChatCompletionClient
from spoox.environment import Environment
from spoox.interface import Interface


class AgentSystem(ABC):
    """
    This class serves as the base for all agent system implementations.
    It provides fundamental runtime functionality such as the human-agent interaction loop (see `start()`),
    which requests user input and executes agents in response.
    Additionally, it implements timeout management and comprehensive logging support.

    Concrete agent system implementations just have to implement:

    - `_build_agents()` that initializes all concreate single agents and their autogen message subscriptions.
    - `_trigger_agents()` implementing the triggering of the agent flow given the latest user intput message, typically realized by publishing an autogen message.
    - `get_state()` simple function returning the most important objects within a dictionary that should be logged and could be useful for post-analysis.
    """

    def __init__(self, interface: Interface, model_client: ChatCompletionClient,
                 environment: Environment, timeout: int = 600, logs_dir: Path = Path.cwd()):

        self.interface = interface
        self.model_client = model_client
        self.environment = environment
        self.timeout = timeout

        self.runtime = SingleThreadedAgentRuntime()

        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.logs_dir = logs_dir / f"spoox_logs_{timestamp}"
        self.logs_dir.mkdir(parents=True)
        self.usage_stats = dict()
        self.init_usage_stats()

        # event passed to single agents to notify them when a timeout occurs
        self._timeout_event = asyncio.Event()
        # async function that contains the timeout countdown if started
        self._timeout_countdown = None

    @property
    def timeout_event(self) -> asyncio.Event:
        return self._timeout_event

    async def start(self) -> None:
        """
        Start and run the agent system.
        The agent system is initialized and enters an infinite loop that alternates between
        waiting for user input and executing the agents by calling self._trigger_agents.
        The system exits when the user enters 'q', 'exit', or 'stop'.
        """

        # start the agent system
        await self.environment.start()
        await self._build_agents()
        self.save_logs()
        start_time = time.time()

        # user input and agent calling loop
        while True:

            user_input = self.interface.request_user_input("Query...")
            if user_input in ['q', 'exit', 'stop']:
                break

            self.runtime.start()
            await self._trigger_agents(user_input)
            self._start_timeout_countdown()
            await self.runtime.stop_when_idle()
            self._cancel_timeout_countdown()
            self.save_logs()

        # stop entirely
        await self.environment.stop()
        await self.runtime.close()
        self.save_logs(stopped=True, exec_time_sec=int(time.time() - start_time))

    def init_usage_stats(self) -> None:
        """
        Initializes a shared dictionary for collecting interesting statistics.
        This dictionary is provided to all single agents to enable a centralized collection.
        """

        self.usage_stats['llm_calls_count'] = 0
        self.usage_stats['tool_call_counts'] = dict()
        self.usage_stats['tool_calls'] = []
        self.usage_stats['model_client_exceptions'] = []
        self.usage_stats['agent_errors'] = []
        self.usage_stats['prompt_tokens'] = []
        self.usage_stats['completion_tokens'] = []
        self.usage_stats['next_agent_calling_chain'] = []
        self.usage_stats['group_chat_message_lengths'] = []

    def _cancel_timeout_countdown(self) -> None:
        """Cancels the timeout event without triggering or restarting the timeout countdown."""

        if self._timeout_countdown is not None and not self._timeout_countdown.done():
            self._timeout_countdown.cancel()
        self._timeout_countdown = None

        if self._timeout_event.is_set():
            self._timeout_event.clear()

    def _start_timeout_countdown(self) -> None:
        """
        Starts a timeout countdown for self.timeout seconds. Once the timeout is reached, self.timeout_event is set.
        The self.timeout_event is shared with all individual agents, allowing them to terminate their execution
        gracefully once the event is set. Directly stopping the runtime while agents are still running
        can result in a ValueError being raised by the autogen runtime. To avoid this, an event is used to notify
        all agents to stop as soon as possible instead of abruptly terminating execution.
        """

        async def _timeout():
            await asyncio.sleep(self.timeout)
            error_message = f"Agent system timeout after {self.timeout}s."
            self.interface.print_highlight(error_message, "TimeoutError")
            self.usage_stats["agent_errors"].append(("TimeoutError", error_message))
            self._timeout_event.set()

        self._cancel_timeout_countdown()
        self._timeout_countdown = asyncio.create_task(_timeout())

    def save_logs(self, stopped: bool = False, exec_time_sec: int = 0) -> None:
        """
        Store agent system logs, including usage_stats, get_state dictionary, the entire interface and some meta-data.
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

    @abstractmethod
    async def _build_agents(self) -> None:
        """
        Initializing all agents, including all message subscriptions.
        Must be implemented by the respective agent system implementation.
        """
        pass

    @abstractmethod
    async def _trigger_agents(self, user_input: str) -> None:
        """
        Triggers the execution flow of the agent system's single agents, given the latest user input.
        Must be implemented by the respective agent system implementation.
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """
        Returns the current state of the agent system for logging and later analysis.
        Must be implemented by the respective agent system implementation.
        """
        pass
