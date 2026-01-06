import json
from abc import ABC, abstractmethod

from autogen_core import FunctionCall, CancellationToken, BaseAgent
from autogen_core.models import FunctionExecutionResult
from autogen_core.tools import BaseTool

from spoox.interface import Interface


class Environment(ABC):
    """
    This abstract base class manages a collection of tools and other environmental stuff,
    providing methods to start/stop/reset the environment, retrieve available tools,
    and execute tool calls with logging, error handling, and usage statistics tracking.
    Typically, an agent is equipped with an environment.
    """

    def __init__(self, interface: Interface):
        self.interface = interface
        self.tools = []
        self._started = False
        self.additional_tool_descriptions = ""

    @abstractmethod
    async def start(self):
        """Starts the environment. Should be called once during agent system startup."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the environment. Should be called once during agent system shutdown."""
        pass

    @abstractmethod
    async def reset(self):
        """
        Resets the environment. Typically called when an agent starts working in a shared environment
        to ensure no dependencies exist from previous agent operations.
        """
        pass

    @abstractmethod
    def get_tools(self, agent: BaseAgent) -> list[BaseTool]:
        """Returns a list of tools the agent should be equipped with."""
        pass

    @abstractmethod
    def get_additional_tool_descriptions(self, agent: BaseAgent) -> [str]:
        """
        Agent system prompts may include additional descriptions for their environment and tools.
        This function returns supplementary description text for the specific agent when needed.
        """
        pass

    @abstractmethod
    def _check_tool_call_confirmation(self, call: FunctionCall) -> str:
        """
        Check if user configuration should be seeked for the given FunctionCall.
        Returns an empty string if call is confirmed and can be executed, or a rejection message.
        """
        pass

    async def execute_tool_call(
            self, tools, call: FunctionCall, cancellation_token: CancellationToken, interface: Interface,
            usage_stats: dict, caller_name: str = ""
    ) -> FunctionExecutionResult:
        """
        This method executes a tool call by finding the matching tool by name, running it with the provided arguments.

        Args:
            tools (list[Tool]): List of all available tools callable by the agent.
            call (FunctionCall): FunctionCall to be executed.
            call (FunctionCall): FunctionCall to be executed.
            cancellation_token (CancellationToken): CancellationToken.
            interface (Interface): Interface for user-facing logging.
            usage_stats (dict): Dictionary of usage statistics, provided by the agent system.
            caller_name (str): Agent topic type.

        Returns:
            FunctionExecutionResult: Filled FunctionExecutionResult.
        """

        # logging
        args_parsed = json.loads(call.arguments)
        tool_name = call.name
        interface.print_tool_call(
            f"**Tool**: {tool_name}  \n**Arguments**:  \n{args_parsed}  \n",
            f"{caller_name} - tool_call"
        )

        if not self._started:
            raise RuntimeError(f"Environment must be started. Make sure `.start()` is called.")

        # find tool by name and run it
        tool = next((tool for tool in tools if tool.name == call.name), None)
        if tool is None:
            feResult = FunctionExecutionResult(call_id=call.id, content=f"Tool '{tool_name}' is not known.",
                                               is_error=True, name=call.name)

        else:
            # seek user confirmation
            rejection_message = self._check_tool_call_confirmation(call)
            if rejection_message != "":
                feResult = FunctionExecutionResult(call_id=call.id,
                                                   content=f"Tool call was rejected due to: (UserMessage) {rejection_message}.",
                                                   is_error=True, name=call.name)

            else:
                # run the tool and capture the result
                try:
                    result = await tool.run_json(args_parsed, cancellation_token)
                    feResult = FunctionExecutionResult(call_id=call.id, content=tool.return_value_as_string(result),
                                                       is_error=False, name=tool.name)
                except Exception as e:
                    feResult = FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)

        # logging
        interface.print_tool_call(
            f"**Tool**: {feResult.name}  \n**Is error**: {feResult.is_error}  \n**Content**:  \n{feResult.content}  \n",
            f"{caller_name} - tool_call_result"
        )
        usage_stats['tool_calls'].append((call, feResult))
        if tool_name in usage_stats['tool_call_counts']:
            usage_stats['tool_call_counts'][tool_name] += 1
        else:
            usage_stats['tool_call_counts'][tool_name] = 1

        return feResult
