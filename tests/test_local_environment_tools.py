from spoox.environment.local_environment import LocalEnvironment

def test_get_tools_for_singleton_agent():
    # Lightweight subclass to bypass heavy __init__
    class DummyLE(LocalEnvironment):
        def __init__(self):
            pass
        def _check_tool_call_confirmation(self, agent):
            return []
    le = DummyLE()
    le._shell_tool = type('T', (), {'name': 'shell'})()
    le._python_tool = type('T', (), {'name': 'python'})()
    le._terminal_tool = type('T', (), {'name': 'terminal'})()
    le._search_tool = type('T', (), {'name': 'search'})()
    class SingletonAgent:
        pass
    agent = SingletonAgent()
    tools = le.get_tools(agent)
    assert isinstance(tools, list)
    assert [t.name for t in tools] == ['shell', 'python', 'terminal', 'search']
