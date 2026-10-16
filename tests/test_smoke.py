"""Smoke test: the package imports and exposes a version."""


def test_import():
    import fsl

    assert fsl.__version__
