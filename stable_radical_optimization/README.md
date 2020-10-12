# Stable radical optimization w/ RL

## Run the optimization

submit via
```bash
sbatch --output=/scratch/$USER/rlmolecule/slurm.%j.out submit_combined.sh
```

## Optimization with monitoring

Clone repo on Eagle:

```bash
cd /scratch/$USER
git clone git@github.com:NREL/rlmolecule.git
```

Configure the optimization job by editing `stable_rad_config.py`.  At a minimum, 
modify `config.sql_basename` to use your own namespace; relevant tables will be
created at runtime if they don't already exist.  Note that all configuration 
parameters in the `alphazero.config` can be modified here, too, and the changes
will propagate everywhere when the job is run.

Run the job initialization to create postgres tables with your namespace.  This 
only needs to be done once if you're using the same namespace throughout your
experiments.

```bash
conda activate /projects/rlmolecule/pstjohn/envs/tf2_[cpu,gpu]
cd /scratch/$USER/rlmolecule/stable_radical_optimization
python initialize.py
```

Start the remote monitoring notebook.  A common choice for this is to use 
a DAV node.  But be mindful that these are shared nodes;  in particular, if you 
run tensorflow, make sure you [limit GPU memory growth](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)! (The
notebook here doesn't use tensorflow).

```bash
# SSH to DAV node
ssh $USER@ed[1-3,5-6].hpc.nrel.gov
cd /scratch/$USER/rlmolecule/

# Create screen session (allows your session to live on if you log out)
screen -S jup
# ctrl-a d   # detach from session
# ctrl-a -r jup    # re-attach to session

# Activate env and start notebook
conda activate /projects/rlmolecule/pstjohn/envs/tf2_[cpu,gpu]
jupyter notebook --no-browser --ip=0.0.0.0

# Note the port number used by the notebook
```

Create SSH tunnel from local machine to remote host running notebook

```bash
ssh -NL 8765:localhost:[port] $USER@[notebook-hostname]
# Here, [port] is whatever port jupyter connected to on the remote host (see previous step),
# and [notebook-hostname] is the machine where the notebook is running.
```

Connect to the notebook from your browser via `localhost:8765`, and open 
`stable_radical_optimization/monitor_progress.ipynb`.  This is a live notebook 
that points to the relevant tables.

Run the optimization

```bash
sbatch --output=/scratch/$USER/rlmolecule/slurm.%j.out submit_combined.sh
```

Postgres tables will begin to be populated; run the monitoring notebook to check
the status of the search.
