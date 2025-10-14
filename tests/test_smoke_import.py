def test_import_and_version():
    import codeless_orchestrator as co  # noqa: F401

    assert hasattr(co, "__version__"), "__version__ not exposed"
    assert isinstance(co.__version__, str) and co.__version__, "__version__ must be non-empty"

