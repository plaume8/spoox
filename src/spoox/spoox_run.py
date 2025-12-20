import argparse
import asyncio

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment.LocalEnvironment import LocalEnvironment
from spoox.utils import setup_model_client, setup_agent_system

nest_asyncio.apply()

from spoox.interface.CLInterface import CLInterface

"""
example usage:
python src/spoox/spoox.py -m gpt-5-mini -a mas-group-chat-m -l False -d False -e False
"""

if __name__ == "__main__":
    """Simple runner for any agents - static setup"""

    parser = argparse.ArgumentParser(description="Agentu argument parser")
    parser.add_argument("-m", "--model-id", required=False, default='gpt-5-mini',
                        help="Model id (str)")
    parser.add_argument("-a", "--agent-id", required=False, default="singleton", help="Agent id (str)")
    parser.add_argument("-l", "--logging", required=False, default=False,
                        help="Show detailed logs, default set to false (bool)")
    parser.add_argument("-d", "--in-docker", required=False, default=False,
                        help="Should be set to true if called within a docker container and using Ollama model (bool)")

    args = parser.parse_args()
    model_id = str(args.model_id)
    agent_id = str(args.agent_id)
    logging = str(args.logging).lower() in ("yes", "true", "t", "y", "1")
    in_docker = str(args.in_docker).lower() in ("yes", "true", "t", "y", "1")

    load_dotenv()

    # setup model client
    model_client = setup_model_client(model_id=model_id, docker_access=in_docker)

    # setup environment and interface
    environment = LocalEnvironment()
    interface = CLInterface(logging_active=logging)

    # setup agent system
    agent = setup_agent_system(agent_id, model_client, environment, interface, logs_dir='/tmp/spoox')
    try:
        asyncio.run(agent.start())
    except Exception as e:
        interface.print(str(e), f"Exception during agent system execution.")
