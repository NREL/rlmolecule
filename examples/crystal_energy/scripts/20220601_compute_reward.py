#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

tqdm.pandas()

print(np.__version__)
print(pd.__version__)

from rlmolecule.crystal import utils, crystal_reward, reward_utils


# ## Compute the new reward for the relaxed structures

# In[6]:


# Dataframe containing competing phases from NRELMatDB
print("Reading inputs/competing_phases.csv")
df_competing_phases = pd.read_csv('inputs/competing_phases.csv')
print(f"\t{len(df_competing_phases)} lines")
print(df_competing_phases.head(2))


# In[13]:


from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PDEntry
from rlmolecule.crystal.ehull import fere_entries


# In[8]:


df_competing_phases['energy'] = (
    df_competing_phases.energyperatom *
    df_competing_phases.sortedformula.apply(lambda x: Composition(x).num_atoms)
)
# convert the dataframe to a list of PDEntries used to create the convex hull
pd_entries = df_competing_phases.apply(
    lambda row: PDEntry(Composition(row.sortedformula),
                        row.energy),
    axis=1
)
print(f"\t{len(pd_entries)} entries")
competing_phases = pd.concat([pd.Series(fere_entries), pd_entries]).reset_index()[0]


# In[9]:


relaxed_structures_file = "/projects/rlmolecule/jlaw/inputs/structures/battery/battery_relaxed_structures.json.gz"
relaxed_structures = utils.read_structures_file(relaxed_structures_file)
df = pd.read_csv("/projects/rlmolecule/jlaw/inputs/structures/battery/battery_relaxed_energies.csv")
df.head(2)

df = df.set_index("id")
df['structure'] = pd.Series(relaxed_structures)
df.head(2)

df['cond_ion'] = df['composition'].apply(lambda comp: reward_utils.get_conducting_ion(Composition(comp)))
df2 = df[df.cond_ion.isin([Element(e) for e in ['Li', 'Na', 'K']])]
print(f"{len(df2)} / {len(df)} structures have Li, Na, or K as the conducting ion")

rewarder = crystal_reward.StructureRewardBattInterface(competing_phases)
rewards = df.progress_apply(
    lambda row: rewarder.compute_reward(
        row.structure,
        row.energyperatom,
        row.index,
    ),
    axis=1
)

rewards = pd.DataFrame([[strc_id, reward, data]
                       for strc_id, (reward, data) in rewards.iteritems()],
                       columns=["id", "reward", "data"],
                       )

out_file = "outputs/20220609_rewards/batt_relaxed_rewards.csv"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
print(f"writing {out_file}")
rewards.to_csv(out_file)



# ## use dask
#from dask.distributed import Client
#from dask_jobqueue import SLURMCluster
#import dask.dataframe as dd
#
#
#def setup_dask_client(n_nodes=2, n_processes=36, debug=False):
#    ###cluster objects
##     n_processes = 36  # number of processes to run on each node
#    memory = 90000  # to fit on a standard node; ask for 184,000 for a bigmem node
#    walltime = '30' if debug else '180'
#    queue = 'debug' if debug else None
#    
#    cluster = SLURMCluster(
#        project='rlmolecule',
#        walltime='30' if debug else '180',  # 30 minutes to fit in the debug queue; 180 to fit in short
#        job_mem=str(memory),
#        job_cpu=36,
#        interface='ib0',
#        local_directory='/tmp/scratch/dask-worker-space',
#        cores=36,
#        processes=n_processes,
#        memory='{}MB'.format(memory),
#        extra=['--lifetime-stagger', '60m'],
#        queue='debug' if debug else None  # 'debug' is limited to a single job -- comment this out for larger runs
#    )
#
#    print(cluster.job_script())
#
#    #create a client
#    dask_client = Client(cluster)
#
#    # scale cluster
#    n_nodes = 1 if debug else n_nodes
#    cluster.scale(n_processes * n_nodes)
#    return dask_client, cluster
#
#
## make the structures and energies into a dask dataframe
## df_rel.set_index('id', inplace=True)
## df_rel['structure'] = pd.Series(rel_structures)
#df_dask = dd.from_pandas(df[:1000], chunksize=10)
#
#### Dask
## now use dask to compute the decomposition energy
#dask_client, cluster = setup_dask_client(debug=True)
#
#
#def compute_reward(structure, energyperatom, strc_id):
#    global rewarder, competing_phases
#    if rewarder is None:
#        rewarder = crystal_reward.StructureRewardBattInterface(competing_phases)
#    return rewarder.compute_reward(structure, 
#                                   energyperatom, 
#                                   strc_id)
#
#
#
#results = df_dask.map_partitions(
#        lambda df_x: df_x.apply(
#               lambda row: compute_reward(
#                   row.structure,
#                   row.energyperatom,
#                   row.index,
#                ),
#            axis=1),
#        meta=(pd.Series(dtype=object))
#)
#
#out = results.compute()
#out
#
## #df_out = pd.DataFrame({'decomp_energy': pd.Series(finished)})
## out_file = predicted_energies_file.replace('.csv', '_decomp_energy.csv')
## print(out_file)
## S_out.to_csv(out_file)


