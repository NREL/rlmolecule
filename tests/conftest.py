import tempfile

import pytest


@pytest.fixture(scope="class")
def tmpdirname():
    """
    A directory for the checkpoint files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
