from tempfile import NamedTemporaryFile

import pytest
from sqlalchemy import create_engine


@pytest.fixture(scope='class')
def engine():
    # Tensorflow's `from_generator` tends to cause issues with in-memory sqlite databases due to threading,
    # so here (and likely in other small codes) we'll want to make sure we at least write to a local file.
    with NamedTemporaryFile(suffix='.db') as file:
        engine = create_engine(f'sqlite:///{file.name}',
                               connect_args={'check_same_thread': False},
                               execution_options={"isolation_level": "AUTOCOMMIT"}
                               )
        yield engine
        engine.dispose()
