from enum import Enum
from pathlib import Path
from typing import Optional, Any

from autogen_core import BaseAgent, FunctionCall
from autogen_core.tools import BaseTool
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools import DuckDuckGoSearchResults

from spoox.environment import Environment
from spoox.environment.code_executors import CodeExecutorLocal
from spoox.environment.tools import ShellTool
from spoox.environment.tools import TerminalTool
from spoox.interface import Interface


class ConfirmationMode(Enum):
    """The level of confirmation the agent will seek from the user before interacting with the environment."""

    STRICT = 'strict'  # Agent seeks user confirmation before each environment interaction.
    SELF_EVALUATION = 'self_evaluation'  # Agent autonomously decides when confirmation is needed (e.g., `cd` does not require confirmation, while `rm` commands always do).
    NO_CONFIRMATION = 'no_confirmation'  # Agent never seeks user confirmation. Warning: Use only in isolated or sandboxed environments.


class LocalEnvironment(Environment):
    """
    This is a concrete environment implementation providing basic programming tools and access the local computer.
    Also, `get_additional_tool_descriptions` and `get_tools` are configured for each relevant specific spoox agent.
    """

    def __init__(self, interface: Interface, confirmation_mode: ConfirmationMode,
                 work_dir: Optional[Path] = None, user: Optional[str] = None):
        super().__init__(interface)

        self.confirmation_mode = confirmation_mode
        self._code_executor = CodeExecutorLocal(work_dir=work_dir, user=user)

        self._shell_tool = ShellTool(self._code_executor)
        self._terminal_tool = TerminalTool()
        self._python_tool = PythonCodeExecutionTool(self._code_executor)
        self._search_tool = LangChainToolAdapter(DuckDuckGoSearchResults(output_format="list"))

    async def start(self):
        """Stop the environment. Should be called once during agent system shutdown."""
        await self._code_executor.start()
        self._started = True

    async def stop(self):
        """Stop the environment. Should be called once during agent system shutdown."""
        await self._code_executor.stop()
        await self._terminal_tool.stop()
        self._started = False

    async def reset(self):
        """
        Resets the environment. Typically called when an agent starts working in a shared environment
        to ensure no dependencies exist from previous agent operations.
        """
        await self._code_executor.restart()
        await self._terminal_tool.reset()

    def get_tools(self, agent: BaseAgent) -> list[BaseTool]:
        """Returns a list of tools the agent should be equipped with. All relevant spoox agents are represented."""

        class_name = agent.__class__.__name__
        if class_name in ["SingletonAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool, self._search_tool]
        elif class_name in ["ExplorerAgent"]:
            return [self._shell_tool, self._search_tool, self._terminal_tool]
        elif class_name in ["SolverAgent", "SubTaskSolverAgent", "RefinerAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool, self._search_tool]
        elif class_name in ["TesterAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool]
        # default: return no tools (e.g. Approver, Summarizer, ...)
        return []

    def get_additional_tool_descriptions(self, agent: BaseAgent) -> [str]:
        """
        Agent system prompts may include additional descriptions for their environment and tools.
        All relevant spoox agents are represented.
        """

        class_name = agent.__class__.__name__
        shell_descr = f"""- Use Shell tool for simple, single bash commands. Do **not** use it for commands that open interactive terminal programs (e.g. git log or man)."""
        py_descr = f"""- Use PythonExecutor tool for complex logic, scripting, or data processing."""
        terminal_descr = f"""- Use Terminal tool to execute commands in a persistent terminal instance and get back the exact terminal screen. This is especially useful for interactive programs such as git log, man, or vim."""

        if class_name in ["SingletonAgent"]:
            return [shell_descr, py_descr, terminal_descr]
        elif class_name in ["ExplorerAgent"]:
            return [shell_descr, terminal_descr]
        elif class_name in ["SolverAgent", "SubTaskSolverAgent", "RefinerAgent"]:
            return [shell_descr, py_descr, terminal_descr]
        elif class_name in ["TesterAgent"]:
            return [shell_descr, py_descr, terminal_descr]
        # default: return no additional tool descriptions (e.g. Approver, Summarizer, ...)
        return []

    def _check_tool_call_confirmation(self, call: FunctionCall) -> str:
        """
        Check if user configuration should be seeked for the given FunctionCall.
        Returns an empty string if call is confirmed and can be executed, or a rejection message.
        For the LocalEnvironment this is dependent on the selected confirmation_mode.
        """
        restricted_tool_names = ["shell", "python", "terminal", "codeexecutor"]
        confirmation_choices = ['confirm', 'reject']

        # NO confirmation required
        if call.name.lower() not in restricted_tool_names:
            return ""
        if self.confirmation_mode == ConfirmationMode.NO_CONFIRMATION:
            return ""

        # confirmation required
        user_choice = 'reject'
        if self.confirmation_mode == ConfirmationMode.STRICT:
            user_choice = self.interface.request_select_choice(
                f"Please confirm whether to proceed with the latest {call.name} tool call:",
                confirmation_choices
            )
        elif self.confirmation_mode == ConfirmationMode.SELF_EVALUATION:
            if self._risk_in_code_execution(call.arguments):
                user_choice = self.interface.request_select_choice(
                    f"Please confirm whether to proceed with the latest {call.name} tool call:",
                    confirmation_choices
                )
            else:
                user_choice = 'confirm'
        if user_choice == 'reject':
            return self.interface.request_user_input(
                "Please explain why you chose to reject this tool call:",
                default="Command poses a potential security risk."
            )
        return ""

    def _risk_in_code_execution(self, code: str) -> bool:
        """
        Risk assessment if code snippet can be executed without user confirmation.
        Returns true if risk detected and user confirmation is required.
        """
        # todo better LLM implementation (already added to git issues)
        if any(r in code for r in self.RISK_KEYWORDS):
            return True
        return False

    RISK_KEYWORDS = [
        # --- File system / destructive ---
        "rm ", "rm -", "rm -rf", "unlink", "shutil.rmtree",
        "os.remove", "os.unlink", "os.rmdir",
        "shred", "dd if=", "mkfs", "wipefs",
        # --- Process / execution ---
        "exec(", "eval(", "compile(",
        "subprocess.", "os.system", "os.popen",
        "Popen(", "call(", "run(",
        "bash -c", "sh -c",
        # --- Privilege escalation ---
        "sudo", "su ", "setuid", "setgid",
        # --- Networking / remote code ---
        "curl ", "wget ", "ftp ", "scp ", "rsync ",
        "requests.get", "requests.post",
        "socket.", "paramiko",
        # --- Persistence / system modification ---
        "crontab", "/etc/passwd", "/etc/shadow",
        "systemctl", "service ",
        "chkconfig",
        "~/.bashrc", "~/.profile",
        # --- Disk / hardware ---
        "/dev/sd", "/dev/nvme", "mount ", "umount ",
        # --- Containers / virtualization ---
        "docker run", "docker exec", "docker rm",
        "kubectl", "helm ",
        # --- Environment manipulation ---
        "export ", "unset ",
        "os.environ",
        # --- Potential fork bombs / abuse ---
        ":(){", "fork(", "multiprocessing"
    ]
