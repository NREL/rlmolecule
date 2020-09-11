## Getting started with Redis

### Prerequisites

```
conda env create -f environment.yml 
conda activate redis_integration
```

### Basic Example

Notebook `redis-basic.ipynb` includes the code needed to write/read data into/from Redis. 
The Redis instance being used is run on `app-test.hpc.nrel.gov` (port 6378).
The hostname and the port being used are read from: `redis.json`.

Briefly, here are some examples of interacting with Redis from Python code (from this notebook):
```
rconn = redis.Redis(host=config["host"], port=config["port"])

# Read
key = "...some string..."
rconn.get(key)

# Write
rconn.set(key, "...new value...")
```

For printing all keys/values in the specified namespace, e.g., `rlmol`, run:
```
for key in rconn.keys("rlmol:*"):
    print(key, rconn.get(key))
```

### More Advanced Example: Integrating Eagle jobs with Redis + Orchestration

*Note:* before running Eagle jobs using these scripts, make sure that the directory `outerr` exists in the current directory
(job output and error files will be saved there).

Scripts `test-*` provide a complete, general example of using Redis for job accounting and storing key/value pairs within each job.

* `test_launch.sh` -- Script that submits the job script `test_jobscript.sbatch` to the SLURM queue. 
It gets the job ID assigned by SLURM (e.g., 4226674) and adds this ID 
to the sets maintained by Redis: `<prefix>_ALL` and `<prefix>_PENDING_AND_RUNNING`.
`<prefix>` here is coming from `redis.config` (variable `job_tag`) -- it can be used as a tag, marking all "rlmolecule" jobs 
or a large group of specific experiments (e.g., "rlmol-q4"); for convenience, this tag is set in `redis.config` and
all other scripts "source" this file (the value is eventually passed to the Python script `test_process.py` via an environment variable).
Usage:
```
bash test_launch.sh 
```
   
* `test_jobscript.sbatch` -- Job script that sets job execution params (using `#SBATCH`), activates necessary conda environment, set the aforementioned 
job prefix/tag, and runs `test_process.py`.

* `test_process.py` -- Python script that is supposed to do the bulk of computational work. Currently, the example script reads in 
`redis.json` (and uses `host` and `port` to connect to Redis; instead of `namespace` from the file, it uses the prefix/tag passed through the
environment variable). Similar to the aforementioned notebook `redis-basic.ipynb`, this script writes some random keys/values into Redis using:
```
rconn.set("%s:%s" % (PREFIX, key), value)
``` 
The code also provides an example of reading all keys with numerical values generated within the current job, sorting them, and printing the top three. 

The rest of the scripts aren't strictly needed to manage the keys/values, but are helpful in job accounting and orchestration. Each script is 
very simple and attempts to perform a single, straightforward task:

* `test_jobtracker.sh` -- Meant to run as a background daemon (using something like: `bash test_jobtracker.sh > /dev/null 2>&1`).
This script checks the status of jobs that are in `<prefix>_PENDING_AND_RUNNING`; depending on the current status
reported by SLURM, it will move the job IDs from `<prefix>_PENDING_AND_RUNNING` to: `<prefix>_COMPLETED`, `<prefix>_FAILED`, `<prefix>_TIMEOUT`,
or `<prefix>_CANCELLED` (atomic move between the sets is used within Redis). 
Then the script goes to sleep and keeps repeating its loop until the process is killed. 
This script can be started before running experiments or at a later time to sort out past jobs.

* `test_lastjob.sh` -- Prints all keys/values produced by the last (most recent) job from `<prefix>_ALL`. 
Usage and sample output:
```
$ bash test_lastjob.sh
Key:  rlmol_exp-4238610:bd291298-f442-11ea-ab10-0cc47af587f7
Value: 0.5227367576701759
---
Key:  rlmol_exp-4238610:bd295cbc-f442-11ea-ab10-0cc47af587f7
Value: 0.16931495241029793
---
Key:  rlmol_exp-4238610:bd2931d8-f442-11ea-ab10-0cc47af587f7
Value: 0.6513745153655051
---
```

* `test_status.sh` -- Prints all job IDs from all the job lists mentioned above: "ALL", "PENDING_AND_RUNNING", "COMPLETED", etc. to see the complete picture 
of recent experiments.
Usage and sample output:
```
$ bash test_status.sh 
Jobs under rlmol_exp_ALL:
1) "4226674"
2) "4226676"
3) "4226677"
...
---
Jobs under rlmol_exp_PENDING_AND_RUNNING:
(empty array)
---
Jobs under rlmol_exp_COMPLETED:
1) "4226674"
2) "4226676"
3) "4226677"
...
---
Jobs under rlmol_exp_FAILED:
(empty array)
---
Jobs under rlmol_exp_CANCELLED:
1) "4226679"
2) "4226681"
---
Jobs under rlmol_exp_TIMEOUT:
(empty array)
---
```

* `test_jobcleanup.sh` -- Empties all job ID lists, allowing to start over the accounting. It doesn't remove the produced keys/values though (maybe it can be added as an optional action here in the future).
  
#### Key Features of This Orchestration
* Keys/values produced by one job are *not* mixed with the keys/values from another job. This will allow in the analysis phase to track how the rewards change from job to job as a result of algorithm improvements, if desired. If this separation of keys/values isn't required, the scripts can be trivially updated to use the same prefix/namespace and then they will be mixed in a single large set.
* The Python script being run (`test_process.py`) isn't cluttered with Redis stuff. The prefix is already figured out in `test_jobscript.sh` and passed here through an environment variable, and the Python code just needs to include `rconn.set(...)` and `rconn.get(...)` where necessary.
* This orchestration appears to be general enough to be useful in a number of scenarios and projects.

