import gzip

import ujson
from pymatgen.core import Structure


def read_structures_file(structures_file):
    print(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = ujson.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    print(f"\t{len(structures)} structures read")
    return structures


def write_structures_file(structures_file, structures_dict, round_float=None):
    """ Write pymatgen structures to a gzipped json file
    *round_float*: round floats in json file to the specified number
    """

    def round_floats(o):
        if isinstance(o, float): return round(o, round_float)
        if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
        return o

    if round_float is not None:
        structures_dict = round_floats(structures_dict)

    print(f"writing {structures_file}")
    with gzip.open(structures_file, 'w') as out:
        out.write(ujson.dumps(structures_dict).encode())
