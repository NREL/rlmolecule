Official documentation here:  https://docs.ray.io/en/latest/cluster/cloud.html

The example that currently works uses a cluster launched by the local `example-full.yaml`.
It uses 1 head and 2 worker CPU m5.large instances running the 
`rayproject/ray-ml:latest-cpu` image.  It also installs rllib as part of the 
`setup_commands`.

## Default cluster configuration fails in multiple ways

The default yaml file, which uses a GPU Docker image and compute instance, 
fails in multiple ways "as is".  

1.  Subnet error:

    ```
    No usable subnets found, try manually creating an instance in your specified 
    region to populate the list of subnets and trying this again. Note that the 
    subnet must map public IPs on instance launch unless you set `use_internal_ips: true` 
    in the `provider` config.
    ```
    __Fix:__ Set `use_internal_ips: True` in the `provider` configuration

2. Credentials error:

    ```
    botocore.exceptions.NoCredentialsError: Unable to locate credentials
    ```
    __Fix:__ Set the profile name to `default` in  `~/.aws/credentials`.

3. Disk space error:

    ```
    latest-gpu: Pulling from rayproject/ray-ml
    e4ca327ec0e7: Pull complete 
    ...
    850f7f4138ca: Extracting  3.752GB/3.752GB
    3b7026c2a927: Download complete 
    failed to register layer: Error processing tar file(exit status 1): write /usr/lib/x86_64-linux-gnu/dri/iris_dri.so: no space left on device
    Shared connection to 172.18.106.160 closed.
    New status: update-failed
    !!!
    SSH command failed.
    !!!
    
    Failed to setup head node.
    ```
    __Fix:__ Increasing storage from 100 GiB to 500 GiB seems to have done the trick,
    
    ```
    available_node_types:
        ray.head.default:
            node_config:
                BlockDeviceMappings:
                  - DeviceName: /dev/sda1
                    Ebs:
                         VolumeSize: 500
    ```

## Multi-node training example using only CPU instances

Here are the steps that demonstrated multi-node training of a PPO policy on 
the CartPole-v0 environment.  The Docker image and EC2 instances are cpu-only.

1. Start the cluster using the local version of the yaml file.

    ```
    ray up example-full.yaml
    ```

3. SSH into head node and check ray status.

    ```
    ray attach example-full.yaml  # on local machine
    ray status                    # on remote machine
    ```
    
    Be patient -- the worker nodes take long time to start up and connect to the head!

4. Run a simple training example, ensuring that more than a single node is used. 
With 1 head and 2 worker m5.large nodes, this command runs rllib using all 
available CPUs (there are 6 total).

    ```
    rllib train --run PPO --env CartPole-v0 --ray-num-cpus 6 --config '{"num_workers": 5}' 
    ```

## Multi-node training example using GPU head and CPU worker instances

Same as above, but set

```
docker:
    head_image: "rayproject/ray-ml:latest-gpu"
    worker_image: "rayproject/ray-ml:latest-cpu"
```

and use something like this command to train:

```
rllib train --env CartPole-v0 --run PPO --ray-num-gpu 1 --ray-num-cpu 6 --config '{"num_workers": 5}'
```

Here we are using 1 head GPU instance and 2 worker CPU instances, with a total of 1 GPU and 6 CPUs.


## Tearing down the cluster

To destroy the cluster and attached volumes:

```
ray down example-full.yaml
```

This command releases all resources used by the job (and billing should stop, 
as well).

To _temporarily stop_ the ray cluster:

```
    ray stop example-full.yaml
```

Note that this does not destroy any volumes that are attached -- these persist 
in AWS (and we continue to be billed hourly for their use).


## Hints / Pitfalls 

1. If you're having trouble starting / connecting to a ray cluster, try the
`--no-config-cache` option for your ray command.

2. Check out the EC2 instance types that are compatible with the AMI: 
https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y .  You may need to 
cross-reference this with the instance types that support GPU's:  
https://aws.amazon.com/ec2/instance-types/

3. Private ECR's will not work out of the box.  Use public if possible.

4. If all else fails, check the AWS dashboard to see what's happening and to 
manually stop/terminate instances and volumes.  __Make sure you are looking in the
correct region, where your instances are actually running!__



## Other notes

* Out of two attempts to create ray clusters, one had a failed worker node (1 
out of 2 worker nodes failed, unsure why).

* The cluster is quite slow to launch, 15+ minutes with only 2 small worker nodes. 
This is not just the docker pull / launch steps, but setting up the Ray cluster
itself.  Could be due to using low-performance resources?

