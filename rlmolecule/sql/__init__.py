import hashlib
import io
from collections import OrderedDict

import numpy as np
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
Session = sessionmaker()


def serialize_ordered_numpy_dict(data: {str: np.ndarray}) -> bytes:
    """ Save a dictionary of numpy arrays to a deterministic bytes string

    :param data: a dictionary of numpy arrays
    :return: a bytes string representing the data
    """
    with io.BytesIO() as f:
        np.savez_compressed(f, **OrderedDict(sorted(data.items())))
        return f.getvalue()


def load_numpy_dict(serialized_data: bytes) -> {str: np.ndarray}:
    """ Load a dictionary of numpy arrays from a bytes string

    :param serialized_data: a bytes string representing the data
    :return: The deserialized dictionary of numpy arrays
    """
    with io.BytesIO(serialized_data) as f:
        return dict(np.load(f, allow_pickle=True).items())


def digest(binary_data: bytes) -> str:
    """ Return a SHA256 message digest for the given bytes string

    :param binary_data: The binary data to digest
    :return: A length-64 hex digest of the binary data
    """
    m = hashlib.sha256()
    m.update(binary_data)
    return m.hexdigest()
