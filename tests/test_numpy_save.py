import numpy as np

from rlmolecule.sql import digest, load_numpy_dict, serialize_ordered_numpy_dict


def test_save_ordered_numpy_dict():
    data = {'a': np.ones((5, 3)), 'b': 3 * np.ones((2, 7))}
    data2 = {'b': 3 * np.ones((2, 7)), 'a': np.ones((5, 3))}

    digest_a = digest(serialize_ordered_numpy_dict(data))
    digest_b = digest(serialize_ordered_numpy_dict(data2))

    assert digest_a == digest_b


def test_load():
    data = {'a': np.ones((5, 3)), 'b': 3 * np.ones((2, 7))}

    saved_data = serialize_ordered_numpy_dict(data)
    loaded_data = load_numpy_dict(saved_data)

    for key in data.keys():
        assert np.all(data[key] == loaded_data[key])
