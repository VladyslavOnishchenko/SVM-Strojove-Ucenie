"""Konfigurácia pytest pre celý projekt."""
import sys
import os

import pytest

# Pridá koreňový adresár projektu do sys.path
sys.path.insert(0, os.path.dirname(__file__))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Spustí pomalé testy (napr. GridSearchCV auto-tune)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: označuje testy ako pomalé (GridSearchCV, atď.)")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="potrebuje --run-slow na spustenie")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
