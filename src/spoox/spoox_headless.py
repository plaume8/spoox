import argparse
import asyncio
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment.local_environment import LocalEnvironment
from spoox.interface.log_interface import LogInterface
from spoox.utils import setup_model_client, setup_agent_system

nest_asyncio.apply()

"""
example usage:
python src/spoox/spoox_headless.py -m gpt-5-mini -a singleton -t "create an empty file named dodo in the current dir"
"""


def main() -> None:
    """Entry point for the spoox CLI."""

    parser = argparse.ArgumentParser(description="Spoox argument parser")
    parser.add_argument(
        "-m",
        "--model-id",
        required=False,
        default="gpt-5-mini",
        help="Model id (str)",
    )
    parser.add_argument(
        "-a",
        "--agent-id",
        required=False,
        default="singleton",
        help="Agent id (str)",
    )
    parser.add_argument(
        "-r",
        "--print-reasoning",
        required=False,
        default=True,
        help="Print reasoning process in terminal, default set to true (bool)",
    )
    parser.add_argument(
        "-d",
        "--in-docker",
        required=False,
        default=False,
        help="Should be set to true if called within a docker container and using Ollama model (bool)",
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        help="Task description (str)",
    )
    parser.add_argument(
        "-l",
        "--logs-dir",
        required=False,
        default="/tmp/spoox",
        help="Logs dir path (str)",
    )
    parser.add_argument(
        "-x",
        "--max-timeout",
        required=False,
        default=3600,  # 60min default max
        help="Max timeout in seconds (int)",
    )

    args = parser.parse_args()

    model_id = str(args.model_id)
    agent_id = str(args.agent_id)
    print_reasoning = str(args.print_reasoning).lower() in ("yes", "true", "t", "y", "1")
    in_docker = str(args.in_docker).lower() in ("yes", "true", "t", "y", "1")
    task = str(args.task)
    logs_dir = Path(str(args.logs_dir))
    max_timeout = int(args.max_timeout)

    load_dotenv()

    # setup model client
    model_client = setup_model_client(model_id=model_id, docker_access=in_docker)

    # setup environment and interface
    environment = LocalEnvironment()

    # setup headless interface -> using log interface
    interface = LogInterface(logging_active=True, print_live=print_reasoning)
    interface.user_delegate.user_input = [task, 'q']
    interface.user_delegate.default_user_choice = 'confirm'

    # setup and run agent system
    agent = setup_agent_system(agent_id, model_client, environment, interface, max_timeout, logs_dir)
    try:
        asyncio.run(agent.start())
    except Exception as e:
        interface.print_highlight(str(e), f"Exception during agent system execution.")


if __name__ == "__main__":
    main()
