Official documentation here:  https://docs.ray.io/en/latest/cluster/cloud.html

The example that currently works is expressed by the `example-full.yaml` file.
It uses 1 head and 2 worker CPU m5.large instances running the 
`rayproject/ray-ml:latest-cpu` image.  The default yaml file, which uses a 
GPU Docker image and compute instance, fails in multiple ways "as is"

* You have to set `use_internal_ips: True` in the `provider` configuration, and

* Starting a ray cluster using the default, GPU-enabled Docker image crashes on 
a "no space left on disk" error.  (__TODO:__  Find a fix for this!).

Here are the steps that demonstrated multi-node training of a PPO policy on 
the CartPole-v0 environment using only CPUs.

1. Create `~/.aws/credentials` and set profile name to `[default]` (rather than
what is auto-generated in the AWS console).  If you're using AWS for the first 
time, the simplest thing to do is manually create this file and paste in the 
following data.

        [default]
        aws_access_key_id=...
        aws_secret_access_key=...
        aws_session_token=...

Note that the profile name needs to be set to `[default]` for the remaining 
steps to work out-of-the-box because this is the credential profile that 
`boto3` looks for by default.

2. Start the cluster.  Note that the yaml file has been modified slightly to
pip install rllib as part of the `setup_commands`.

        ray up example-full.yaml

3. SSH into head node and check ray status.

        ray attach example-full.yaml  # locally
        ray status  # remote

4. Run a simple training example, ensuring that more than a single node is used. 
With 1 head and 2 worker m5.large nodes, this command runs rllib using all 
available CPUs (there are 6 total).

        rllib train --run PPO --env CartPole-v0 --ray-num-cpus 6 --config '{"num_workers": 5}' 

Notes:

* Out of two attempts to create ray clusters, one had a failed worker node (1 
out of 2 worker nodes failed, unsure why).

* The cluster is quite slow to launch, 15+ minutes with only 2 small worker nodes. 
This is not just the docker pull / launch steps, but setting up the Ray cluster
itself.  Could be due to using low-performance resources?

