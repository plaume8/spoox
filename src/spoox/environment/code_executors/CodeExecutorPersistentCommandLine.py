import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, ClassVar, Union

from autogen_core import CancellationToken, Component
from autogen_core.code_executor import CodeExecutor, CodeBlock, CodeResult
from pydantic import BaseModel


class CodeExecutorPersistentCommandLineConfig(BaseModel):
    """Configuration for PermanentCommandLineCodeExecutor"""

    timeout: int = 90
    work_dir: Optional[str] = None


class CodeExecutorPersistentCommandLine(CodeExecutor, Component[CodeExecutorPersistentCommandLineConfig]):
    component_config_schema = CodeExecutorPersistentCommandLineConfig

    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "bash",
        "shell",
        "sh",
    ]

    def __init__(self, timeout: int = 90, work_dir: Optional[Union[Path, str]] = Path.cwd()):
        super().__init__()

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")
        self.timeout = timeout

        self._started = False
        self._work_dir = work_dir
        self._code_dir = None
        self._shell_exec = None
        self._runs_count = 0

    @property
    def runs_count(self) -> int:
        """Counter of executed code blocks."""
        return self._runs_count

    async def execute_code_blocks(self, code_blocks: List[CodeBlock],
                                  cancellation_token: CancellationToken) -> CodeResult:

        if len(code_blocks) != 1:
            raise RuntimeError(f"CodeExecutorPersistentCommandLine `code_blocks` must exactly contain one code block.")
        code_block = code_blocks[0]

        if not self._started:
            raise RuntimeError(f"CodeExecutorPersistentCommandLine must be started. Make sure `.start()` is called.")

        if code_block.language.lower() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{code_block.language}' not supported by CodeExecutorPersistentCommandLine.")

        # write code block to the sh file
        code_block_file = Path(f"{self._code_dir.name}/code_block_{self._runs_count}.sh")
        code_block_file.write_text(f"#!/bin/bash\n{code_block.code}\n")
        code_block_file.chmod(0o755)

        # exec code file
        try:
            result = subprocess.run(
                [str(code_block_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            std_aggr = "command execution failed - timeout error"
            return CodeResult(1, std_aggr)

        # format exec output
        if result.stdout and result.stderr:
            return CodeResult(result.returncode, f"<stdout>\n{result.stdout}\n</stdout>\n<stderr>\n{result.stderr}\n</stderr>")
        elif result.stdout:
            return CodeResult(result.returncode, result.stdout)
        elif result.stderr:
            return CodeResult(result.returncode, result.stderr)
        return CodeResult(result.returncode, "")

    async def start(self) -> None:
        if not self._started:
            self._code_dir = tempfile.TemporaryDirectory(dir=self._work_dir)
            self._setup_shell_exec()
            self._runs_count = 0
            self._started = True

    async def stop(self) -> None:
        if self._started:
            self._code_dir.cleanup()
            self._shell_exec.terminate()
            self._started = False

    async def restart(self) -> None:
        if self._started:
            await self.stop()
        await self.start()

    def _setup_shell_exec(self):
        self._shell_exec = subprocess.Popen(
            ['/bin/bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self._work_dir
        )
