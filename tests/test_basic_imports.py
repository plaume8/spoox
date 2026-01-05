import spoox

def test_import_spoox_submodules():
    # Import subpackages to ensure package layout is loadable
    import spoox.environment
    import spoox.agents
    import spoox.utils
    # Basic sanity: module names should resolve to proper packages
    assert spoox.environment.__name__ == "spoox.environment"
    assert spoox.agents.__name__ == "spoox.agents"
    assert spoox.utils.__name__ == "spoox.utils"
