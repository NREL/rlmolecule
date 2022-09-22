# Running QED in AWS Ray Clusters

This page provides a brief summary of launching and using AWS Ray clusters for running QED script available in this repository.
It is based on the instructions from Ray's official documentation for laynching AWS clusters: [https://docs.ray.io/en/latest/cluster/cloud.html](https://docs.ray.io/en/master/cluster/vms/user-guides/launching-clusters/aws.html#install-ray-cluster-launcher)

Below, the instructions prefaced with `*(local)*` are the commands that need to be run in the terminal 
from the directory with the `qed.yml` configuration file (i.e. the directory with this README file). This will likely 
be on the workstation where this repository is cloned and is meant to be used to launch AWS clusters (thus, on this workstation, `~/.aws/credentials` file
needs to be updated for accessing the propoer AWS account). 

This is in contast with the commands prefaced with `*(ray head)*` which will need to be run on the cloud instances that act as Ray head nodes after the clusters are successfully launched.

### `*(local)*` Getting started and launching an AWS Ray cluster:

`# ray up qed.yml --no-config-cache`

After confirming that you indeed want to launch cloud instances and following the configuration output (it can take some time), a cluster will be launched 
with one head node and one worker node. Both will be of [g3.4xlarge](https://aws.amazon.com/about-aws/whats-new/2018/10/introducing-a-new-size-for-amazon-ec2-g3-graphics-accelerated-instances/) instance type, 
with 1 GPU and 16 vCPUs each. 

**Warning:** these are not the smallest/cheapest AWS instances and it is highly advised to terminate them when they are not used
(see the termination instructions provided later on this page).

### `*(local)*` Connect to the launched cluster:

`# ray attach qed.yml --no-config-cache`

After the cluster is launched and configured successfully, you can connect to its head node using this command (behind the scenes, it will establish an ssh connection).
This will change the terminal's prompt to indicate that the following commands will be run on that particular cloud instance.

### `*(ray head)*` Head node operations:

To see info about the cluster's current state:

`# ray status`

There is a period of time, when the worker node is going to be listed under `Pending` in the output. Wait until it moves to the `Healthy` section, and then you
will know that your cluster is fully configured. Keep in mind the process of setting up the worker node will likely take as much time as launching of head node under the previous step.

For troubleshooting, run the following command to follow along the steps in the configuration process:

`# tail -f /tmp/ray/session_latest/logs/monitor.out`

### `*(ray head)*` Run QED example

TBD

Describe an alternative way, with launching it from local environment

### `*(local)*` Access Tensorboard

TBD

### `*(local)*` Access Ray's dashboard

TBD

### `*(local)*` Terminate the cluster

TBD

### Common pitfalls

TBD
