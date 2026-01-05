import json
import os
import shlex
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.context import AgentContext
from pydantic import BaseModel


class ExecInput(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None


_AGENT_ID = "spoox-m"  # "singleton",'spoox-s','spoox-m','spoox-l'
_AGENT_ID_CHAR = "m"  # "singleton",'s','m','l'
_MODEL_CLIENT_ID = "openai"  # "ollama", "openai", "anthropic"
_MODEL_ID = "gpt-5-mini"  # "gpt-oss:20b","qwen3:14b","claude-sonnet-4-5","magistral:24b","gpt-5","gpt-5-mini","gpt-5-nano"

# harbor automatically copies the /logs/agent directory to local Harbor logs;
# as a result, spoox logs are automatically copied as well.
_DOCKER_LOGS_DIR = "/logs/agent/spoox"
_AGENT_MAX_TIMEOUT = 60 * 60 * 2  # 2h


class Spoox(BaseInstalledAgent):
    """
    This script defines a Harbor agent wrapper for Spoox that configures the agent with a specific model.
    Compliant with https://harborframework.com/docs/agents#installed-agents.

    run with: `harbor run -d terminal-bench@2.0 -a /benchmarks/harbor/spoox`
    """

    @staticmethod
    def name() -> str:
        return f"spoox-{_AGENT_ID_CHAR}"

    @property
    def _install_agent_template_path(self) -> Path:
        """
        Path to the jinja template script for installing the spoox agent in the container.
        """
        return Path(__file__).parent / "install_spoox.sh"

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create the commands to run the agent in the container.
        For running spoox, the pre-configured venv is activated and the spoox headless command is executed.
        """
        load_dotenv()
        openai_api_key = str(os.environ.get("OPENAI_API_KEY"))
        safe_instruction = shlex.quote(instruction)
        confirmation_mode = 'confirmation'
        cmd = (f". /opt/venv/bin/activate && spoox-h "
               f"-c {_MODEL_CLIENT_ID} -m {_MODEL_ID} -a {_AGENT_ID} -l {_DOCKER_LOGS_DIR} -x {_AGENT_MAX_TIMEOUT} "
               f"-s {confirmation_mode} -t {safe_instruction}")
        return [ExecInput(command=cmd, env={"OPENAI_API_KEY": openai_api_key})]

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the context with the results of the agent execution. Assumes the run() method has already been called.
        Parses token statistics and adds general agent configs to metadata.
        """
        meta_data = self._load_spoox_logs()
        if meta_data is None:
            return
        context.n_input_tokens = meta_data.get("model-client-total-usage-prompt-tokens")
        context.n_output_tokens = meta_data.get("model-client-total-usage-completion-tokens")
        meta_data['spoox-agent-id'] = _AGENT_ID
        meta_data['spoox-agent-id-char'] = _AGENT_ID_CHAR
        meta_data['model-client-id'] = _MODEL_CLIENT_ID
        meta_data['model-id'] = _MODEL_ID
        context.metadata = meta_data

    def _load_spoox_logs(self) -> Optional[dict]:
        """Find and load spoox logs."""
        try:
            # find spoox logs
            spoox_dir = self.logs_dir / "spoox"
            spoox_logs_dirs = [e for e in spoox_dir.iterdir() if e.name.startswith("spoox_")]
            if len(spoox_logs_dirs) != 1:
                return None
            # load metadata logs
            meta_data_logs = spoox_logs_dirs[0] / "meta_data.json"
            if not meta_data_logs.is_file():
                return None
            with meta_data_logs.open("r", encoding="utf-8") as f:
                meta_data = json.load(f)
            if isinstance(meta_data, dict):
                return meta_data
        except Exception:
            # log parsing is non-critical
            pass
        return None
