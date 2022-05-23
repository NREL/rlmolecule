""" Compute the decomposition energy of any structure based on its composition and (predicted) total energy
"""
# example call:
# python scripts/compute_decomp_energy.py \
#    --relaxed-energies-file /projects/rlmolecule/pgorai/mcts_validation/mctsvalidation_Mg80decor_1.csv \
#    --out-file /projects/rlmolecule/jlaw/crystals/2022-01-25/mctsvalidation_Mg80decor_1.csv \
#    --write-stab-calcs

import argparse
from pathlib import Path
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import gzip
import json
import re
import numpy as np
import sqlalchemy
import pandas as pd
import nfp
from pymatgen.core import Structure

sys.path.append('../../')
from rlmolecule.crystal import utils
sys.path.insert(0, "")  # make sure the current directory is read
from scripts import nrelmatdbtaps
from scripts import stability
from scripts import ehull


def main(relaxed_energies_file,
         out_file,
         comp_phases_file,
         write_cvex_inputs=False,
         write_stab_calcs=False,
         ):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if write_stab_calcs:
        cvex_hull_dir = str(out_file).replace('.csv', '-stab-analysis')
        os.makedirs(cvex_hull_dir, exist_ok=True)

    print(f"reading {relaxed_energies_file}")
    df_rel = pd.read_csv(relaxed_energies_file)
    print(df_rel.head(2))
    strc_energies = dict(zip(df_rel['id'], df_rel['energyperatom']))

    print(f"reading {comp_phases_file}")
    df_phases = pd.read_csv(comp_phases_file)
    print(df_phases.head(2))

    print(f"Computing decomposition energy for {len(strc_energies)} structures.")
    print(f"Writing to {out_file}")
    with open(out_file, 'w') as out:
        out.write(','.join(["id", "energyperatom", "decomp_energy"]) + '\n')
        strc_hull_nrgy = {}
        for strc_id, energy in tqdm(strc_energies.items()):
            #try:
            comp = strc_id.split('_')[0]
            cvex_hull_file = f"{cvex_hull_dir}/{strc_id}.txt" if write_stab_calcs else None
            inputs_file = f"{cvex_hull_dir}/{strc_id}_inputs.txt" if write_cvex_inputs else None
            decomp_energy = ehull.convex_hull_stability(comp,
                                                        energy,
                                                        df_phases,
                                                        inputs_file=inputs_file,
                                                        out_file=cvex_hull_file)
            print(strc_id, energy, decomp_energy)
            #except:
            #    print(f"Failed for {strc_id}. Skipping")
            #    continue
            strc_hull_nrgy[strc_id] = decomp_energy
            out.write(f"{strc_id},{energy},{decomp_energy}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute the decomposition energy of any structure based on '
                    'its composition and (predicted) total energy')
    parser.add_argument('--relaxed-energies-file', '-r',
                        type=Path,
                        required=True,
                        help="CSV with at least two columns titled 'id' and 'energyperatom'")
    parser.add_argument('--out-file', '-o',
                        type=Path,
                        required=True,
                        help="Output CSV containing computed decomposition energy")
    parser.add_argument('--comp-phases-file',
                        type=Path,
                        default='/projects/rlmolecule/jlaw/rlmolecule/examples/crystal_energy/inputs/competing_phases.csv',
                        help="Competing phases file necessary for constructing the convex hull")
    parser.add_argument('--write-cvex-inputs',
                        action='store_true',
                        help="Write the inputs to the convex hull analysis of each composition "
                        "to a dir with the same name as <out-file>, "
                        " with '-stab-analysis' appended to it")
    parser.add_argument('--write-stab-calcs',
                        action='store_true',
                        help="Write the stability analysis of each decoration "
                        "to a dir with the same name as <out-file>, "
                        " with '-stab-analysis' appended to it")
    
    args = parser.parse_args()
    main(args.relaxed_energies_file,
         args.out_file,
         args.comp_phases_file,
         args.write_cvex_inputs,
         args.write_stab_calcs)
