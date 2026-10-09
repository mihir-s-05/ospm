import pytest

from ospm.benchmarks import load_project
from ospm.memory import Memory


@pytest.fixture(scope="session")
def project():
    return load_project()


@pytest.fixture
def memory():
    m = Memory(":memory:")
    yield m
    m.close()
