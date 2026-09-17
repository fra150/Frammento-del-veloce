"""Conftest: sys.path + flag --run-slow per i test marcati slow."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="esegue anche i test marcati slow (demo completa con 7 figure)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="test slow: rieseguire con --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
