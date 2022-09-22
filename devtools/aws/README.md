# Running QED in AWS Ray Clusters

This page provides a brief summary of launching and using AWS Ray clusters for running QED script available in this repository.
It is based on the instructions from Ray's official documentation for lauching AWS clusters: [https://docs.ray.io/en/latest/cluster/cloud.html](https://docs.ray.io/en/master/cluster/vms/user-guides/launching-clusters/aws.html#install-ray-cluster-launcher)

### Notation in the instructions and where different commands need to run

Below, the instructions prefaced with `*(local)*` are the commands that need to be run in the terminal 
from the directory with the `qed.yml` configuration file (i.e. the directory with this README file). This will likely 
be on the workstation where this repository is cloned and is meant to be used to launch AWS clusters (thus, on this workstation, `~/.aws/credentials` file
needs to be updated for accessing the proper AWS account). 

This is in contrast with the commands prefaced with `*(ray head)*` which will need to be run on the cloud instances that act as Ray head nodes after the clusters are successfully launched.

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

### `*(ray head)*` Run QED example:

Before proceeding with this step, make sure that all launched instances are configured (i.e., `Healthy`, as described under the previous step).

The code from this repo should be already cloned on the head node (that is part of the configuration process defined in `qed.yml` file). Run the QED example:

`# python ~/rlmolecule/examples/benchmarks/qed/run_qed_aws.py`

The output should include a series of tables with the column named: `episode_reward_max`. The value under that column should increase as the training progresses and eventually reach the value around 0.93.

**Alternatively**, you can run the same command without being on the head node (`*(local)*`, as with the rest of the commands) by using:

`# ray exec qed.yml 'python ~/rlmolecule/examples/benchmarks/qed/run_qed_aws.py' --no-config-cache`

With either `*(local)*` or `*(ray head)*` version of this command, you would want to leave it running for some time to see reasonable rewards (and explore the results through TensorBoard, as described below). 

### `*(local)*` Access TensorBoard:

For this step, you might want to use a separate terminal window. To see the produced results through TensorBoard dashboard, run (this will start the dashboard on the head node and also set up the port forwarding for port 6006):

`# ray exec qed.yml 'tensorboard --logdir=/home/ray/ray_results --port 6006 --bind_all' -p 6006 --no-config-cache`

Leave this terminal window to continue running this command and open your browser to navigate to: `localhost:6006`. That should display TensorBoard and show the latest training results for QED. 

### `*(local)*` Terminate the cluster

`# ray down qed.yml`
