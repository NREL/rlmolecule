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

4. Subnets problem (causes an intermittent problem with ssh connections timing out)
    
    For every sandbox account, NREL adds several public subnets (despite them not being allowed to have any public access) to enable Internet Gateway service, which allows  EC2 instances to access the broader internet (e.g. reach github.com, etc). The EC2 instances need to be given public IPs briefly from those subnet IP pools in order to make public internet requests. 
    
    To see public and private subnets, navigate to VPC within AWS console -> Subnets. Currently there are 2 public and 4 private subnets for this project. It was confirned that the subnet IDs will not change for the lifetime of the sandbox account for a given project.
    
    The problem with Ray clusters (with the default config) comes from the fact that if subnets aren't specified for head and worker nodes, those nodes will be assigned to subnets randomly, which causes this intermittent problem with ssh connectons (intermittent here means that different but identically launched instances will act diffently rather than the same instance acting differently over time).  
    
    The solution here is to pick *one of the private* subnet IDs and use it in the configuration file, as described below. This subnet ID should look like: `subnet-XY...Z`.
    
    This solution also requires specifying a particular security group to be used with the head and worker nodes. You can see available security groups under EC2 console -> Security Groups. After you have tried `ray up...`, you should have a security group with the name: `ray-autoscaler-default`. You should use the corresponding group ID, which looks like: `sg-XY..Z`.
    
    To specify both the subnet and the security group, edit the .yml file and add the entire `NetworkInterfaces:` block as demonstrated below. **Repeat** this under both `ray.head.default:` and `ray.worker.default:` (make sure that exactly the same subnet ID and security group ID are used in both cases).
    
    Note that subnet ID is spelled `SubnetId` here (exact capitalization is important).
    
    ```
    available_node_types:
      ray.head.default:
        resources: {}
            node_config:
            InstanceType: g3.4xlarge
            ImageId: ami-0a2363a9cff180a64 # Deep Learning AMI (Ubuntu) Version 30
            # You can provision additional disk space with a conf as follows
            BlockDeviceMappings:
                - DeviceName: /dev/sda1
                  Ebs:
                      VolumeSize: 600
            # Additional options in the boto docs.
            NetworkInterfaces:
                - DeviceIndex: 0 # Primary network interface.
                  SubnetId: subnet-XY...Z # replace with appropriate subnet (i.e. one of _private_ subnets for the sandbox account being used)
                  Groups:
                    - sg-XY...Z # Replace with appropriate Security Group ID.
    ```
    
    Example of such configuration was found at: 
    https://github.com/ray-project/ray/blob/fcb044d47c9673ddcf97908097f4b38c02d54826/python/ray/autoscaler/aws/example-network-interfaces.yaml
    
    Note that the example configuration available at the link above includes the line `InterfaceType: efa # Use EFA for higher throughput and lower latency` (under under `NetworkInterfaces:` under `ray.worker.efa:`). It does **not** work with the current configuration (under `ray.worker.default:`); maybe it is not available for the selected instance type in the selected region (for more info, see a comment about EFA in the linked example).

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

## Enabling Tensorboard and connecting to it

Similar to the examples above, you can run rllib example but this time you can specify the destination of the produced training results using `--local-dir LOCAL_DIR`, for instance:
      
```
rllib train --run PPO --env CartPole-v0 --ray-num-cpus 6 --config '{"num_workers": 5}' --local-dir /app/rlmolecule/output
```

While this comamnd is running, run this command in a separate terminal window (from the directory with the `example-full.yml` file that was used to launch the cluster):

```
ray exec example-full.yaml 'tensorboard --logdir=/app/rlmolecule/output --port 6006 --bind_all' -p 6006 --no-config-cache
```
This will run Tensorboard on the head node at port 6006 and will also set up the necessary ssh port forwarding for that port, which will allow seeing the dashboard locally. Good ouput from this command should end with the line: `TensorBoard 2.7.0 at http://<hostname>:6006/ (Press CTRL+C to quit)`.

Then, navigate to: `http://localhost:6006` in your browser. **Important:** it seems to work fine in Chrome, but in Safari, this will just show an empy page (it is possible that the browser settings do not allow JavaScript, and they need to be udapted).

## Ray dashboard

Ray comes with its own dashboard showing the available workers, used/available memory, CPUs and GPUs, etc. This dashboard runs at port 8265.

To connect to it, run this command (from the directory with the `example-full.yml` file that was used to launch the cluster):
```
ray dashboard example-full.yaml --no-config-cache
```

Then, keep that command running in a terminal and open the dashboard at: `http://localhost:8265`. You can use Chrome or Safari (or, possibly, other browsers) to see this dashboard.

## Hints / Pitfalls 

1. If you're having trouble starting / connecting to a ray cluster, try the
`--no-config-cache` option for your ray command.

1. Running `ray down ...` does not necessarily terminate your instance.  In some
cases, they are only stopped (I couldn't reproduce this).  See:
https://github.com/ray-project/ray/issues/6207.  There is probably a configuration
option somewhere to override...


1. Check out the EC2 instance types that are compatible with the AMI: 
https://aws.amazon.com/marketplace/pp/prodview-x5nivojpquy6y .  You may need to 
cross-reference this with the instance types that support GPU's:  
https://aws.amazon.com/ec2/instance-types/

1. Private ECR's will not work out of the box.  Use public if possible.

1. If all else fails, check the AWS dashboard to see what's happening and to 
manually stop/terminate instances and volumes.  __Make sure you are looking in the
correct region, where your instances are actually running!__  

1. It might end up being the case that a single, beefy GPU node will give better
performance than the head/worker paradigm.


## Other notes

* Out of two attempts to create ray clusters, one had a failed worker node (1 
out of 2 worker nodes failed, unsure why).

* The cluster is quite slow to launch, 15+ minutes with only 2 small worker nodes. 
This is not just the docker pull / launch steps, but setting up the Ray cluster
itself.  Could be due to using low-performance resources?

