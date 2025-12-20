import tempfile
import uuid
from pathlib import Path
from typing import List, ClassVar

from autogen_core import CancellationToken, Component
from autogen_core.code_executor import CodeExecutor, CodeBlock, CodeResult
from pydantic import BaseModel
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession


_TMUX_SESSION: dict[str, TmuxSession] = {}


def set_code_executor_tmux_session(key: str, session: TmuxSession) -> None:
    global _TMUX_SESSION
    _TMUX_SESSION[key] = session


def get_code_executor_tmux_session(key: str) -> TmuxSession:
    if key not in _TMUX_SESSION:
        raise ValueError(f"TerminalTool: no global Tmux session set for key {key}.")
    return _TMUX_SESSION[key]


class CodeExecutorTerminalBenchConfig(BaseModel):
    """Configuration for CodeExecutorTerminalBenchConfig"""

    tmux_session_key: str
    timeout: int = 90


class CodeExecutorTerminalBench(CodeExecutor, Component[CodeExecutorTerminalBenchConfig]):
    component_config_schema = CodeExecutorTerminalBenchConfig

    SUPPORTED_LANGUAGES: ClassVar[List[str]] = [
        "python",
        "bash",
    ]

    def __init__(self, tmux_session_key: str, timeout: int = 90):
        super().__init__()

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")
        self.timeout = timeout

        self._tmux_session_key = tmux_session_key
        self._session = get_code_executor_tmux_session(tmux_session_key)
        self._started = False
        self._runs_count = 0
        self._local_code_dir = None
        self._local_code_dir_path = None
        self._container_code_dir_path = None

    @property
    def runs_count(self) -> int:
        """Counter of executed code blocks."""
        return self._runs_count

    async def execute_code_blocks(self, code_blocks: List[CodeBlock],
                                  cancellation_token: CancellationToken = None) -> CodeResult:

        if len(code_blocks) != 1:
            raise RuntimeError(f"CodeExecutorTerminalBench `code_blocks` must exactly contain one code block.")
        code_block = code_blocks[0]

        if not self._started:
            raise RuntimeError(f"CodeExecutorTerminalBench must be started. Make sure `.start()` is called.")

        if code_block.language.lower() not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{code_block.language}' not supported by CodeExecutorTerminalBench.")

        # setup code block language dependents
        type_of_file = ".py" if code_block.language.lower() == "python" else ".sh"
        caller_keyword = "python" if code_block.language.lower() == "python" else "bash"

        # write code block to the py or sh file and copy to TB docker container
        rand_id = uuid.uuid4().hex[:8]
        code_block_file = self._local_code_dir_path / f"code_block_{self._runs_count}_{rand_id}{type_of_file}"
        code_block_file.write_text(f"{code_block.code}\n")
        code_block_file.chmod(0o755)
        self._session.copy_to_container(
            code_block_file,
            container_dir=str(self._container_code_dir_path),
        )

        # run python file in container
        terminal_out, exit_code = self._send_command_get_output(
            f"{caller_keyword} {self._container_code_dir_path}/{code_block_file.name}"
        )

        # remove first and last line so that agent could not get confused
        # first line just contains calling the code file
        # last line which is just the typical input mask line
        terminal_out_lines = terminal_out.splitlines()
        if len(terminal_out_lines) >= 2:
            terminal_out_lines = terminal_out_lines[1:-1]
        terminal_out = "\n".join(terminal_out_lines)

        return CodeResult(exit_code, terminal_out)  # todo exit code 0 is not valid for all outputs...

    def _send_command_get_output(self, command: str) -> (str, int):
        """
        We can only retrieve the entire shell output, not just the last commands results.
        Therefore, we echo a marker before running the command to know where to cut.
        """
        marker = f"__MARKER-{self._runs_count}-{uuid.uuid4().hex[:8]}__"
        self._runs_count += 1
        self._session.send_command(TerminalCommand(command=f"echo '{marker}'", block=True))
        try:
            # adding `echo` ensures that we always have 3 lines
            # and can cut off the first and last one without removing the output
            self._session.send_command(
                TerminalCommand(command=f"{command} ; echo", max_timeout_sec=self.timeout, block=True))
        except TimeoutError:
            return f"Timeout error after {self.timeout} seconds. Be aware that bash commands which open interactive terminal interfaces (e.g. 'top', 'man' or 'vim') can also result in a timeout.", 124
        terminal_out = self._session.capture_pane(capture_entire=True)
        terminal_out_cleaned = terminal_out.split(marker)[-1].strip()
        return terminal_out_cleaned, -1

    async def start(self) -> None:
        if not self._started:
            # setup local temp code dir
            self._local_code_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
            self._local_code_dir_path = Path(self._local_code_dir.name)
            # setup temp code dir in container
            self._container_code_dir_path = Path(f"/opt/{self._local_code_dir_path.name}")
            self._send_command_get_output(f"mkdir {self._container_code_dir_path}")
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

