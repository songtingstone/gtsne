def test_import():
    import gtsne

    assert gtsne.__version__ is not None
    assert gtsne.__version__ != "0.0.0"
    assert len(gtsne.__version__) > 0


def test_import_gtsne():
    from gtsne import gtsne  # noqa
