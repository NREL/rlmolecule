
## Generate action graph
- The file `icsd_prototypes_lt50atoms_lt100dist.json.gz` has all ICSD prototype structures matching the composition types we're interested with two constraints:
    - < 50 atoms
    - max distance between atoms < 100 after normalizing so the minimum distance is 1. These are problematic structures that have atoms in the same location
Example call to build the action space:
```
python scripts/build_action_space.py \
    --prototypes-json ../../rlmolecule/crystal/inputs/icsd_prototypes_lt50atoms_lt100dist.json.gz \
    --comp-file ../../rlmolecule/crystal/inputs/compositions.csv.gz \
    --out-pref ../../rlmolecule/crystal/inputs/lt50atoms_lt100dist \
    --write-proto-json
```

### Requirements
The only additional package (PyPolyhedron) required to run the phase-stability code in the tensorflow environment can be installed from here, https://github.com/frssp/PyPolyhedron
