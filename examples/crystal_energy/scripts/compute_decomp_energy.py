import os
import sys
from collections import defaultdict
from tqdm import tqdm
import gzip
import json
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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    context='paper',
    font_scale=8/8.8,
#     context="talk",
    style='ticks',
    color_codes=True,
    rc={'legend.frameon': False})

plt.rcParams['svg.fonttype'] = 'none'


def read_structures_file(structures_file):
    print(f"reading {structures_file}")
    with gzip.open(structures_file, 'r') as f:
        structures_dict = json.loads(f.read().decode())
    structures = {}
    for key, structure_dict in structures_dict.items():
        structures[key] = Structure.from_dict(structure_dict)
    print(f"\t{len(structures)} structures read")
    return structures


def convex_hull_stability(df_competing_phases, structure: Structure, predicted_energy):
    strc = structure

    # Add the new composition and the predicted energy to "df" if DFT energy already not present
    comp = strc.composition.reduced_composition.alphabetical_formula.replace(' ','')

    df = df_competing_phases
    if comp not in df.reduced_composition.tolist():
        df = df_competing_phases.append({'sortedformula': comp, 'energyperatom': predicted_energy, 'reduced_composition': comp}, ignore_index=True)

    # Create a list of elements in the composition
    ele = strc.composition.chemical_system.split('-')

    # Create input file for stability analysis 
    inputs = nrelmatdbtaps.create_input_DFT(ele, df, chempot='ferev2')

    # Run stability function (args: input filename, composition)
    stable_state = stability.run_stability(inputs, comp)
    if stable_state == 'UNSTABLE':
        stoic = ehull.frac_stoic(comp)
        hull_nrg = ehull.unstable_nrg(stoic, comp, inputs)
        #print("energy above hull of this UNSTABLE phase is", hull_nrg, "eV/atom")
    elif stable_state == 'STABLE':
        stoic = ehull.frac_stoic(comp)
        hull_nrg = ehull.stable_nrg(stoic, comp, inputs)
        #print("energy above hull of this STABLE phase is", hull_nrg, "eV/atom")
    else:
        print(f"ERR: unrecognized stable_state: '{stable_state}'.")
        print(f"\tcomp: {comp}")
        return None
    return hull_nrg




# load the relaxed structures and run the hull energy code
relaxed_energies_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/battery_relaxed_energies.csv"
print(f"reading {relaxed_energies_file}")
df_rel = pd.read_csv(relaxed_energies_file)
print(df_rel.head(2))
strc_energies = dict(zip(df_rel['id'], df_rel['energyperatom']))

comp_phases_file = "/home/jlaw/projects/arpa-e/crystals/rlmolecule/examples/crystal_energy/inputs/competing_phases.csv"
print(f"reading {comp_phases_file}")
df_phases = pd.read_csv(comp_phases_file)
print(df_phases.head(2))

strcs_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/battery_relaxed_structures.json.gz"
rel_structures = utils.read_structures_file(strcs_file)

out_file = "outputs/relaxed-hull-energies.tsv"
print(f"Computing decomposition energy for {len(rel_structures)} structures.")
print(f"Writing to {out_file}")
with open(out_file, 'w') as out:
    strc_hull_nrgy = {}
    for strc_id, strc in tqdm(rel_structures.items()):
        try:
            hull_energy = convex_hull_stability(df_phases, strc, strc_energies[strc_id])
        except:
            print(f"Failed for {strc_id}. Skipping")
            continue
        strc_hull_nrgy[strc_id] = hull_energy
        out.write(f"{strc_id}\t{strc_energies[strc_id]}\t{hull_energy}\n")

