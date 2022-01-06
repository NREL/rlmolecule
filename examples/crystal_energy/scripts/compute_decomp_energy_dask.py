import os
import sys
from tqdm import tqdm
import numpy as np
# import psycopg2
import sqlalchemy
import pandas as pd
from collections import defaultdict
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import dask.dataframe as dd

import nfp
sys.path.append('../../')
import rlmolecule
from rlmolecule.sql.run_config import RunConfig
from rlmolecule.sql import Base, Session
from rlmolecule.sql.tables import GameStore, RewardStore, StateStore
from rlmolecule.crystal import utils
sys.path.insert(0, "")  # make sure the current directory is read
from scripts import nrelmatdbtaps
from scripts import stability
from scripts import ehull


#def convex_hull_stability(df_competing_phases, strc, predicted_energy):
def convex_hull_stability(row, df_competing_phases):
    strc = row.structure
    predicted_energy = row.predicted_energyperatom
    # Add the new composition and the predicted energy to "df" if DFT energy already not present
    comp = strc.composition.reduced_composition.alphabetical_formula.replace(' ','')

    df_cp = df_competing_phases
    if comp not in df_cp.reduced_composition.tolist():
        df_cp = df_competing_phases.append({'sortedformula': comp, 'energyperatom': predicted_energy, 'reduced_composition': comp}, ignore_index=True)

    # Create a list of elements in the composition
    ele = strc.composition.chemical_system.split('-')

    # Create input file for stability analysis
    inputs = nrelmatdbtaps.create_input_DFT(ele, df_cp, chempot='ferev2')
    # if this function failed to create the input, then skip this structure
    if inputs is None:
        #return [row.index, None]
        return

    # Run stability function (args: input filename, composition)
    try:
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
            #return [row.index, None]
            return
    except SystemError as e:
        print(e)
        print(f"Failed at stability.run_stability for {str(row)}. Skipping\n")
        #return [row.index, None]
        return 
    #return [row.index, hull_nrg]
    return hull_nrg


def setup_dask_client(n_nodes=2, n_processes=36, debug=False):
    ###cluster objects
#     n_processes = 36  # number of processes to run on each node
    memory = 90000  # to fit on a standard node; ask for 184,000 for a bigmem node
    walltime = '30' if debug else '180'
    queue = 'debug' if debug else None
    
    cluster = SLURMCluster(
        project='rlmolecule',
        walltime='30' if debug else '180',  # 30 minutes to fit in the debug queue; 180 to fit in short
        job_mem=str(memory),
        job_cpu=36,
        interface='ib0',
        local_directory='/tmp/scratch/dask-worker-space',
        cores=36,
        processes=n_processes,
        memory='{}MB'.format(memory),
        extra=['--lifetime-stagger', '60m'],
        queue='debug' if debug else None  # 'debug' is limited to a single job -- comment this out for larger runs
    )

    print(cluster.job_script())

    #create a client
    dask_client = Client(cluster)

    # scale cluster
    n_nodes = 1 if debug else n_nodes
    cluster.scale(n_processes * n_nodes)
    return dask_client, cluster

# load the relaxed structures and run the hull energy code
# instead, use the predicted values for the decomposition energy
#relaxed_energies_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/battery_relaxed_energies.csv"
#predicted_energies_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/zintl_relaxed_energies.csv"
predicted_energies_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/outputs/icsd_battery_relaxed/hypo_vsad5_icsd_vsad5_seed1/overall_battery_pred_err.csv"
print(f"reading {predicted_energies_file}")
df_rel = pd.read_csv(predicted_energies_file, index_col='id')
#df_rel.rename({'energyperatom': 'predicted_energyperatom'}, axis=1, inplace=True)
print(df_rel.head(2))
#strc_energies = dict(zip(df_rel.index, df_rel['energyperatom']))
#strc_energies = dict(zip(df_rel.index, df_rel['predicted_energyperatom']))
#print(list(strc_energies.items())[:2])

comp_phases_file = "/home/jlaw/projects/arpa-e/crystals/rlmolecule/examples/crystal_energy/inputs/competing_phases.csv"
print(f"reading {comp_phases_file}")
df_phases = pd.read_csv(comp_phases_file)
print(df_phases.head(2))

print(f"{len(df_rel)} entries before adding structures")
strcs_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/battery_relaxed_structures.json.gz"
#strcs_file = "/projects/rlmolecule/jlaw/crystal-gnn-fork/inputs/structures/zintl_relaxed_structures.json.gz"
rel_structures = utils.read_structures_file(strcs_file)
df_rel['structure'] = pd.Series(rel_structures)
print(f"{len(df_rel)} entries after adding structures")
print(df_rel.head(2))

### Dask
# now use dask to compute the decomposition energy
#dask_client, cluster = setup_dask_client(debug=True)
dask_client, cluster = setup_dask_client(n_nodes=3)

# make the structures and energies into a dask dataframe
df_dask = dd.from_pandas(df_rel, chunksize=10)

#results = df_dask.map_partitions(
#        lambda df_x: df_x[['structure', 'predicted_energyperatom']].apply(
#            convex_hull_stability, df_competing_phases=df_phases, axis=1).values,
#        meta=['test', 0])

results = df_dask.map_partitions(
        lambda df_x: df_x[['structure', 'predicted_energyperatom']].apply(
            convex_hull_stability, df_competing_phases=df_phases, axis=1),
        meta=('decomp_energy', float))

S_out = results.compute()

#df_out = pd.DataFrame({'decomp_energy': pd.Series(finished)})
out_file = predicted_energies_file.replace('.csv', '_decomp_energy.csv')
print(out_file)
S_out.to_csv(out_file)
#df_out.to_parquet('/projects/rlmolecule/pstjohn/crystal_fingerprints/fingerprints.parquet')

dask_client.shutdown()

## For sequential computation:
##out_file = "outputs/relaxed-hull-energies.tsv"
##print(f"Computing decomposition energy for {len(rel_structures)} structures.")
##print(f"Writing to {out_file}")
##with open(out_file, 'w') as out:
#strc_hull_nrgy = {}
#for strc_id, strc in tqdm(list(rel_structures.items())[:50]):
#    #try:
#    hull_energy = convex_hull_stability(df_phases, strc, strc_energies[strc_id])
#    #except:
#    #    print(f"Failed for {strc_id}. Skipping")
#    #    continue
#    strc_hull_nrgy[strc_id] = hull_energy
#print(strc_hull_nrgy)
##        out.write(f"{strc_id}\t{strc_energies[strc_id]}\t{hull_energy}\n")


