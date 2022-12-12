import os
import pytest
import sqlite3

from cannier_framework.globals import ARGS, DATA


def pytest_addoption(parser):
    parser.addoption(
        "--schema-file", action="store", dest="schema-file", type=str
    )


@pytest.fixture
def args():
    yield ARGS
    ARGS.__dict__.clear()


@pytest.fixture
def data():
    yield DATA
    DATA.__dict__.clear()
    DATA.__init__()


@pytest.fixture
def db_file(request, tmpdir):
    schema_file = request.config.getoption("schema-file")
    with open(schema_file, "r") as f: schema = f.read()
    db_file = os.path.join(tmpdir.strpath, "db.sqlite3")
    with sqlite3.connect(db_file) as con: con.executescript(schema)
    yield db_file