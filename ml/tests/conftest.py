import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("torch") is None:
        skip = pytest.mark.skip(reason="torch not installed (lite environment)")
        for item in items:
            if "torch" in item.keywords:
                item.add_marker(skip)
