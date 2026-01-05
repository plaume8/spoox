
import spoox

def test_imports_are_available_ext():
    # Import subpackages to ensure package layout is loadable
    import spoox.environment
    import spoox.agents
    import spoox.utils
    assert spoox.environment is not None
    assert spoox.agents is not None
    assert spoox.utils is not None
