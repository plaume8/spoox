from typing import Optional

from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.code_executor import CodeExecutor
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self

from spoox.environment.code_executors.code_executor_local import CodeBlockTimeout
from spoox.environment.tools.utils import output_truncat

"""
For human developers, opening a terminal and executing simple Bash commands serves as the standard lightweight interface 
for performing general server management and diverse OS tasks. The action space is almost unbounded, multiple commands 
can be composed, modified, and chained, to accomplish complex objectives. The Shell tool shall make this powerful 
functionality available to agents. It has intentionally been designed with minimal complexity. To invoke it, the agent 
only needs to provide the command as a string. The command is always executed in the current user’s home directory, 
and the resulting shell output, including the exit code, is stored to the agent’s context. Thereby, the execution 
environment is not persistent. Commands that manipulate the shell session, such as cd, do not affect future tool calls.
"""


class CodeExecutionInput(BaseModel):
    """Input object for ShellTool."""

    command: str = Field(description="The Bash command that should be executed in the shell.")
    # timeout: int = Field(
    #    description="Maximum duration (in seconds) the command may run. "
    #                "Define only if you want to increase the default timeout of 20s. "
    #                "Timeout value must not exceed 120s.",
    #    default=20
    # )


class CodeExecutionResult(BaseModel):
    """Result object for ShellTool."""

    output: str
    exit_code: Optional[int] = None

    @model_serializer
    def ser_model(self) -> str:
        if self.exit_code is None:
            return f"<output> {self.output} </output>"
        return f"<exit-code> {self.exit_code} </exit-code>  \n  <output> {self.output} </output>"


class ShellToolConfig(BaseModel):
    """Configuration for ShellTool."""

    executor: ComponentModel
    output_max: int
    description: str = "Execute a Bash command in the shell, in the users current directory."


class ShellTool(BaseTool[CodeExecutionInput, CodeExecutionResult], Component[ShellToolConfig]):
    """A tool that executes Bash code in a code executor and returns the output."""

    component_config_schema = ShellToolConfig

    def __init__(self, executor: CodeExecutor, output_max: int = 20000):
        super().__init__(
            CodeExecutionInput,
            CodeExecutionResult,
            "Shell",
            "Execute a Bash command in the shell, in the users current directory."
        )
        self._executor = executor
        self._output_max = output_max

    async def run(self, args: CodeExecutionInput, cancellation_token: CancellationToken) -> CodeExecutionResult:
        # execute code
        code_block = CodeBlockTimeout(code=args.command, language="bash", timeout=60)  # args.timeout
        result = await self._executor.execute_code_blocks(
            code_blocks=[code_block], cancellation_token=cancellation_token
        )
        # make sure the output is cut when too long
        output = output_truncat(result.output, self._output_max)
        return CodeExecutionResult(exit_code=result.exit_code, output=output)

    def _to_config(self) -> ShellToolConfig:
        """Convert current instance to config object."""
        return ShellToolConfig(executor=self._executor.dump_component(), output_max=self._output_max)

    @classmethod
    def _from_config(cls, config: ShellToolConfig) -> Self:
        """Create instance from config object."""
        executor = CodeExecutor.load_component(config.executor)
        return cls(executor=executor, output_max=config.output_max)
