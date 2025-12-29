from pathlib import Path
from typing import Optional

from autogen_core import BaseAgent
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools import DuckDuckGoSearchResults

from spoox.environment.Environment import Environment
from spoox.environment.code_executors.CodeExecutorLocal import CodeExecutorLocal
from spoox.environment.tools.ShellTool import ShellTool
from spoox.environment.tools.TerminalTool import TerminalTool


class LocalEnvironment(Environment):

    def __init__(self, work_dir: Optional[Path] = None, user: Optional[str] = None):
        super().__init__()

        self._code_executor = CodeExecutorLocal(work_dir=work_dir, user=user)

        self._shell_tool = ShellTool(self._code_executor)
        self._terminal_tool = TerminalTool()
        self._python_tool = PythonCodeExecutionTool(self._code_executor)
        self._search_tool = LangChainToolAdapter(DuckDuckGoSearchResults(output_format="list"))

    async def start(self):
        await self._code_executor.start()
        self._started = True

    async def stop(self):
        await self._code_executor.stop()
        await self._terminal_tool.stop()
        self._started = False

    async def reset(self):
        await self._code_executor.restart()
        await self._terminal_tool.reset()

    def get_tools(self, agent: BaseAgent):
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

