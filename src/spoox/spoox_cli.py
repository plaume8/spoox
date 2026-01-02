import argparse
import asyncio
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment import LocalEnvironment
from spoox.interface import CLInterface
from spoox.utils import setup_model_client, setup_agent_system


"""
example usage:
python src/spoox/spoox_cli.py -m gpt-5-mini -a spoox-m -l False -d False -e False
"""

nest_asyncio.apply()

LOGS_DIR = Path('/tmp/spoox')
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Entry point for the spoox CLI.
    This CLI script configures and runs an agent system.
    """

    parser = argparse.ArgumentParser(description="Agentu argument parser")
    parser.add_argument(
        "-c",
        "--model-client-id",
        required=False,
        default="openai",
        help="Model client, options: 'ollama', 'openai', 'anthropic'.",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        required=False,
        default='gpt-5-mini',
        help="Model id (str)."
    )
    parser.add_argument(
        "-a",
        "--agent-id",
        required=False,
        default="singleton",
        help="Agent id (e.g. 'singleton', 'spoox-m') (str)."
    )
    parser.add_argument(
        "-l",
        "--logging",
        required=False,
        default=False,
        help="Show detailed logs (bool)."
    )

    args = parser.parse_args()
    client_id = str(args.model_client_id)
    model_id = str(args.model_id)
    agent_id = str(args.agent_id)
    logging = str(args.logging).lower() in ("yes", "true", "t", "y")

    load_dotenv()

    # setup and run agent system
    model_client = setup_model_client(client_id=client_id, model_id=model_id)
    environment = LocalEnvironment()
    interface = CLInterface(logging_active=logging)
    agent = setup_agent_system(agent_id, model_client, environment, interface, logs_dir=LOGS_DIR)
    try:
        asyncio.run(agent.start())
    except Exception as e:
        interface.print(str(e), f"Exception during agent system execution.")


if __name__ == "__main__":
    main()
