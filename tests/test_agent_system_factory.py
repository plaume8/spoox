
from spoox.utils import AgentSystemId, setup_agent_system
from spoox.agents import SingletonAgentSystem

class DummyModelClient:
    pass
class DummyEnvironment:
    pass
class DummyInterface:
    pass


def test_setup_agent_system_returns_instance():
    agent = setup_agent_system(AgentSystemId.SINGLETON, DummyModelClient(), DummyEnvironment(), DummyInterface())
    assert agent is not None
    assert isinstance(agent, SingletonAgentSystem)
