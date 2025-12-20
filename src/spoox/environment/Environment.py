import json
from abc import ABC, abstractmethod

from autogen_core import FunctionCall, CancellationToken
from autogen_core.models import FunctionExecutionResult

from spoox.interface.Interface import Interface

class Environment(ABC):

    def __init__(self):
        self.tools = []
        self._started = False
        self.additional_tool_descriptions = ""

        # special: CallAgentTool is set by the SMASSupervisorAgent
        self.call_agent_tool = None

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def reset(self):
        pass

    @abstractmethod
    def get_tools(self, obj):
        pass

    @abstractmethod
    def get_additional_tool_descriptions(self, obj) -> [str]:
        pass

    async def execute_tool_call(
            self, tools, call: FunctionCall, cancellation_token: CancellationToken, interface: Interface, usage_stats: dict, caller_name: str = ""
    ) -> FunctionExecutionResult:

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
            feResult = FunctionExecutionResult(call_id=call.id, content=f"Tool '{tool_name}' is not known.", is_error=True, name=call.name)
        else:
            # run the tool and capture the result
            try:
                result = await tool.run_json(args_parsed, cancellation_token)
                feResult = FunctionExecutionResult(call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name)
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
