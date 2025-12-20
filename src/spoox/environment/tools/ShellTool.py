from typing import Optional

from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.code_executor import CodeExecutor
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self

from spoox.environment._utils import output_truncat
from spoox.environment.code_executors.CodeExecutorLocal import CodeBlockTimeout

"""
class CodeExecutionInput(BaseModel):
    command: str = Field(description="The Bash command that should be executed in the shell.")
    timeout: int = Field(
        description="Maximum duration (in seconds) the command may run. "
                    "Define only if you want to increase the default timeout of 20s. "
                    "Timeout value must not exceed 120s.",
        default=20
    )
"""


class CodeExecutionInput(BaseModel):
    command: str = Field(description="The Bash command that should be executed in the shell.")


class CodeExecutionResult(BaseModel):
    output: str
    exit_code: Optional[int] = None

    @model_serializer
    def ser_model(self) -> str:
        if self.exit_code is None:
            return f"<output> {self.output} </output>"
        return f"<exit-code> {self.exit_code} </exit-code>  \n  <output> {self.output} </output>"


class ShellToolConfig(BaseModel):
    """Configuration for ShellTool"""

    executor: ComponentModel
    output_max: int
    description: str = "Execute a Bash command in the shell, in the users current directory."


class ShellTool(
    BaseTool[CodeExecutionInput, CodeExecutionResult], Component[ShellToolConfig]
):
    """
    A tool that executes Bash code in a code executor and returns output.

    Args:
        executor: The code executor that will be used to execute the code blocks.
    """

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
        """Convert current instance to config object"""
        return ShellToolConfig(executor=self._executor.dump_component(), output_max=self._output_max)

    @classmethod
    def _from_config(cls, config: ShellToolConfig) -> Self:
        """Create instance from config object"""
        executor = CodeExecutor.load_component(config.executor)
        return cls(executor=executor, output_max=config.output_max)
