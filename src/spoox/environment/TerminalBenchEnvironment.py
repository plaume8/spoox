import uuid

from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools import DuckDuckGoSearchResults
from terminal_bench.terminal.tmux_session import TmuxSession

from spoox.environment.Environment import Environment
from spoox.environment.code_executors.CodeExecutorTerminalBench import CodeExecutorTerminalBench, \
    set_code_executor_tmux_session
from spoox.environment.tools.PythonTool import PythonTool
from spoox.environment.tools.ShellTool import ShellTool
from spoox.environment.tools.TerminalTool import TerminalTool, set_terminal_tmux_session


class TerminalBenchEnvironment(Environment):

    def __init__(self, session: TmuxSession):
        super().__init__()

        session_id = uuid.uuid4().hex[:8]
        set_terminal_tmux_session(session_id, session)
        set_code_executor_tmux_session(session_id, session)

        self._code_executor = CodeExecutorTerminalBench(session_id, timeout=2 * 60)

        self._shell_tool = ShellTool(self._code_executor)
        self._python_tool = PythonTool(self._code_executor)
        self._terminal_tool = TerminalTool(session_id)  # TerminalInteractiveProgramTool(session_id)
        self._search_tool = LangChainToolAdapter(DuckDuckGoSearchResults(output_format="list"))

    async def start(self):
        await self._code_executor.start()
        self._started = True

    async def stop(self):
        await self._code_executor.stop()
        self._started = False

    async def reset(self):
        await self._terminal_tool.reset()
        await self._code_executor.restart()

    def get_tools(self, obj):
        class_name = obj.__class__.__name__
        if class_name in ["SingletonAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool, self._search_tool]
        elif class_name in ["ExplorerAgent"]:
            return [self._shell_tool, self._terminal_tool, self._search_tool]
        elif class_name in ["SolverAgent", "SubTaskSolverAgent", "RefinerAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool, self._search_tool]
        elif class_name in ["TesterAgent"]:
            return [self._shell_tool, self._python_tool, self._terminal_tool]
        elif class_name in ["SMASSupervisorAgent"] and self.call_agent_tool is not None:
            return [self.call_agent_tool]
        # default: return no tools (e.g. Approver, Summarizer, ...)
        return []

    def get_additional_tool_descriptions(self, obj) -> [str]:
        class_name = obj.__class__.__name__
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
