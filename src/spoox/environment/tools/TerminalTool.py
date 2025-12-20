import asyncio
import random

from autogen_core import CancellationToken, Component
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field, model_serializer
from typing_extensions import Self

from spoox.environment.code_executors.TmuxTerminalSession import TmuxTerminalSession


class TerminalInput(BaseModel):
    command: str = Field(description="The command that should be executed in the persistent terminal instance.")
    enter: bool = Field(
        default=True,
        description="Set to True to simulate pressing the Enter key after the command, ensuring it is submitted to the terminal. "
                    "Set to False if the command should only be typed into the terminal without executing Enter. "
                    "Typically, set this to True when submitting Bash commands, and to False when sending tmux keys (e.g., C-b, Tab, Up). "
                    "This field is optional, the default is set to True.")


class TerminalResult(BaseModel):
    running_program: str
    current_screen: str

    @model_serializer
    def ser_model(self) -> str:
        return (f"<running-program> {self.running_program} </running-program>  "
                f"\n  <current-screen>  \n  {self.current_screen}  \n  </current-screen>")


class TerminalToolConfig(BaseModel):
    """Configuration for TerminalTool"""

    tmux_session_name: str
    description: str = (
        "Executes the command in a persistent terminal session, waits for 2 seconds and returns the updated visible terminal screen buffer (size: 128 columns and 16 rows). "
        "Supports both shell commands and interactive commands that follow the tmux tokens format; "
        "tmux token examples: C-a, C-b, C-c, C-l, Tab, Space, Home, End, Insert, Delete, Up, Down, Left, Right, Escape. "
        "Especially ideal for running and controlling interactive terminal programs such as 'git log' or 'vim'."
    )


class TerminalTool(
    BaseTool[TerminalInput, TerminalResult], Component[TerminalToolConfig]
):
    """
    A tool that runs an active terminal can execute commands and returns the entire terminal screen.

    Args:
        tmux_session_name: The terminal bench TmuxSession key.
    """

    component_config_schema = TerminalToolConfig

    def __init__(self, tmux_session_name: str = None):
        super().__init__(
            TerminalInput,
            TerminalResult,
            "Terminal",
            (
                "Executes the command in a persistent terminal session, waits for 2 seconds and returns the updated visible terminal screen buffer (size: 128 columns and 16 rows). "
                "Supports both shell commands and interactive commands that follow the tmux tokens format; "
                "tmux token examples: C-a, C-b, C-c, C-l, Tab, Space, Home, End, Insert, Delete, Up, Down, Left, Right, Escape. "
                "Especially ideal for running and controlling interactive terminal programs such as 'git log' or 'vim'."
            )
        )
        self.tmux_session_name = None
        self._session = None
        self._init_session(tmux_session_name)

    def _init_session(self, tmux_session_name: str = None) -> None:
        """Init a new tmux session - also overrides the current tmux_session_name."""
        if tmux_session_name is None:
            rand_suffix = str(random.randint(10000000, 99999999))
            self.tmux_session_name = f"ts-{rand_suffix}"
        else:
            self.tmux_session_name = tmux_session_name
        self._session = TmuxTerminalSession(session_name=self.tmux_session_name)
        self._session.clear_history()

    async def run(self, args: TerminalInput, cancellation_token: CancellationToken = None) -> TerminalResult:

        if args.enter:
            self._session.send_line(args.command)
        else:
            self._session.send_keys(args.command)

        # before getting the screen -> wait for 2s
        await asyncio.sleep(2)
        current_screen, running_program = self._session.get_screen()
        return TerminalResult(
            running_program=running_program,
            current_screen=current_screen
        )

    async def reset(self):
        if self._session is not None:
            self._session.kill_session()
        self._init_session()

    async def stop(self):
        if self._session is not None:
            self._session.kill_session()

    def _to_config(self) -> TerminalToolConfig:
        """Convert current instance to config object"""
        return TerminalToolConfig(tmux_session_name=self.tmux_session_name)

    @classmethod
    def _from_config(cls, config: TerminalToolConfig) -> Self:
        """Create instance from config object"""
        return cls(tmux_session_name=config.tmux_session_name)
