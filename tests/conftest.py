import tempfile

import pytest

from .sql import engine

@pytest.fixture(scope='class')
def tmpdirname():
    """
    A directory for the checkpoint files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
