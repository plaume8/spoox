import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, ClassVar

from autogen_core import CancellationToken, Component
from autogen_core.code_executor import CodeExecutor, CodeBlock, CodeResult
from pydantic import BaseModel


@dataclass
class CodeBlockTimeout(CodeBlock):
    """A code block extracted from an agent message. Including max timeout."""
    timeout: Optional[int] = None


class CodeExecutorLocalConfig(BaseModel):
    """Configuration for PermanentCommandLineCodeExecutor"""
    timeout_min: Optional[int] = 20
    timeout_max: Optional[int] = 120
    work_dir: Optional[str] = None


class CodeExecutorLocal(CodeExecutor, Component[CodeExecutorLocalConfig]):
    """
    This CodeExecutor executes Python and Bash code blocks locally by writing them to temporary files,
    running them as subprocesses with configurable timeouts and working directories,
    and returning the execution results with stdout/stderr output.
    """

    component_config_schema = CodeExecutorLocalConfig

    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "python",
        "bash",
    ]

    def __init__(self, timeout_min: Optional[int] = 20, timeout_max: Optional[int] = 120,
                 work_dir: Optional[Path] = None, user: Optional[str] = None):
        super().__init__()

        if timeout_min < 1 or timeout_max < 1:
            raise ValueError("Timeout values must be greater than or equal to 1.")
        self.timeout_min = timeout_min
        self.timeout_max = timeout_max

        self._started = False
        self._runs_count = 0
        self._local_code_dir = None
        self._local_code_dir_path = None

        self.work_dir = work_dir or Path.cwd()
        self.user = user or os.environ.get("USER") or os.environ.get("USERNAME")

    @property
    def runs_count(self) -> int:
        """Counter of executed code blocks."""
        return self._runs_count

    async def execute_code_blocks(self, code_blocks: List[CodeBlock],
                                  cancellation_token: CancellationToken) -> CodeResult:
        """
        This method validates and executes a single code block by writing it to a temporary file,
        running it as a subprocess with a bounded timeout, and returning the execution result
        with formatted stdout/stderr output or a timeout error.
        """

        if len(code_blocks) != 1:
            raise RuntimeError(f"CodeExecutorLocal `code_blocks` must exactly contain one code block.")
        code_block = code_blocks[0]

        if not self._started:
            raise RuntimeError(f"CodeExecutorLocal must be started. Make sure `.start()` is called.")

        if code_block.language.lower() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{code_block.language}' not supported by CodeExecutorLocal.")

        # setup code block language dependents
        type_of_file = ".py" if code_block.language.lower() == "python" else ".sh"
        caller_keyword = "python" if code_block.language.lower() == "python" else "bash"

        # write code block to the py or sh file
        rand_id = uuid.uuid4().hex[:8]
        code_block_file = self._local_code_dir_path / f"code_block_{self._runs_count}_{rand_id}{type_of_file}"
        code_block_file.write_text(f"{code_block.code}\n")
        code_block_file.chmod(0o755)

        # extract timeout if available and ensure it stays within defined min max boundaries
        if isinstance(code_block, CodeBlockTimeout) and code_block.timeout is not None:
            timeout = code_block.timeout
            if timeout < self.timeout_min:
                timeout = self.timeout_min
            elif timeout > self.timeout_max:
                timeout = self.timeout_max
        else:
            timeout = self.timeout_min

        # exec code file
        try:
            result = subprocess.run(
                [caller_keyword, str(code_block_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                user=self.user,
                cwd=str(self.work_dir)
            )
        except subprocess.TimeoutExpired:
            return CodeResult(1, f"Command execution failed - timeout of {timeout}s exceeded")

        # format exec output
        if result.stdout and result.stderr:
            return CodeResult(result.returncode,
                              f"<stdout>\n{result.stdout}\n</stdout>\n\n<stderr>\n{result.stderr}\n</stderr>")
        elif result.stdout:
            return CodeResult(result.returncode, result.stdout)
        elif result.stderr:
            return CodeResult(result.returncode, result.stderr)
        return CodeResult(result.returncode, "")

    async def start(self) -> None:
        if not self._started:
            # setup local temp code dir
            self._local_code_dir = tempfile.TemporaryDirectory(dir=Path("/tmp"))
            self._local_code_dir_path = Path(self._local_code_dir.name)
            # others
            self._runs_count = 0
            self._started = True

    async def stop(self) -> None:
        if self._started:
            self._local_code_dir.cleanup()
            self._started = False

    async def restart(self) -> None:
        if self._started:
            await self.stop()
        await self.start()
