from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self

from spoox.environment._utils import output_truncat


class CodeExecutionInput(BaseModel):
    command: str = Field(description="The Bash command that should be executed in the shell.")

# todo open: test if the shell is really persistent


class CodeExecutionResult(BaseModel):
    exit_code: int
    output: str

    @model_serializer
    def ser_model(self) -> str:
        if self.exit_code == -1:
            return f"<shell-output> {self.output} </shell-output>"
        return f"<exit-code> {self.exit_code} </exit-code>  \n  <shell-output> {self.output} </shell-output>"


class PersistentShellToolConfig(BaseModel):
    """Configuration for PersistentShellTool"""

    executor: ComponentModel
    output_max: int
    description: str = "Execute a Bash command in a persistent shell. The shell is persistent therefore commands influence the future state of the shell and future commands."


class PersistentShellTool(
    BaseTool[CodeExecutionInput, CodeExecutionResult], Component[PersistentShellToolConfig]
):
    """
    A tool that executes Bash code in a code executor and returns output.

    Args:
        executor: The code executor that will be used to execute the code blocks.
    """

    component_config_schema = PersistentShellToolConfig

    def __init__(self, executor: CodeExecutor, output_max: int = 20000):
        super().__init__(
            CodeExecutionInput,
            CodeExecutionResult,
            "PersistentShell",
            "Execute a Bash command in a persistent shell. The shell is persistent therefore commands influence the future state of the shell and future commands."
        )
        self._executor = executor
        self._output_max = output_max

    async def run(self, args: CodeExecutionInput, cancellation_token: CancellationToken) -> CodeExecutionResult:
        # execute code
        code_block = CodeBlock(code=args.command, language="bash")
        result = await self._executor.execute_code_blocks(
            code_blocks=[code_block], cancellation_token=cancellation_token
        )
        # make sure the output is cut when too long
        output = output_truncat(result.output, self._output_max)
        return CodeExecutionResult(exit_code=result.exit_code, output=output)

    def _to_config(self) -> PersistentShellToolConfig:
        """Convert current instance to config object"""
        return PersistentShellToolConfig(executor=self._executor.dump_component(), output_max=self._output_max)

    @classmethod
    def _from_config(cls, config: PersistentShellToolConfig) -> Self:
        """Create instance from config object"""
        executor = CodeExecutor.load_component(config.executor)
        return cls(executor=executor, output_max=config.output_max)
