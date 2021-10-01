import argparse
import os

import pandas as pd
from examples.crystal_volume import optimize_crystal_volume as ocv
from pymatgen.core import Composition, Structure
from tqdm import tqdm


def main(structure_file):
    structure_files = []
    print(f"reading {structure_file}")
    if 'POSCAR_' in structure_file:
        strc = Structure.from_file(structure_file)
        structures = [strc]
        structure_files = [structure_file]
    else:
        structures = []
        with open(structure_file, 'r') as f:
            for line in f:
                strc_file = line.rstrip()
                print(f"\treading {strc_file}")
                strc = Structure.from_file(strc_file)
                structures.append(strc)
                structure_files.append(strc_file)

    volume_stats = {}
    for i, strc in tqdm(enumerate(structures)):
        # Compute the volume of the conducting ions.
        conducting_ion_vol, total_vol = ocv.compute_structure_vol(strc)
        frac_conducting_ion_vol = conducting_ion_vol / total_vol if total_vol != 0 else 0
        composition = Composition(strc.formula).reduced_composition
        stats = [str(composition).replace(' ', ''), conducting_ion_vol, total_vol, frac_conducting_ion_vol]
        print(stats)
        strc_file = structure_files[i]
        volume_stats[strc_file] = stats

    df = pd.DataFrame(volume_stats).T
    df.columns = ['composition', 'conducting_ion_vol', 'total_vol', 'fraction']
    print(df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compute the fraction of volume for the conducting ion of a given structure file', )
    parser.add_argument('--structure-file', type=str, help='path/to/POSCAR-file. ' + \
                                                           'Can also give a file containing a list of POSCAR files on which to run')

    args = parser.parse_args()

    if not os.path.isfile(args.structure_file):
        print(f"ERROR: --structure-file '{args.structure_file}' not found")

    main(args.structure_file)
