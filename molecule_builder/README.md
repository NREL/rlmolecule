# Run rollout with multiple CPUs

The file ```run_rollout_workers.sh``` can be used for running multiple instances of rollout on different workers for a **single machine/node** (the procedure has not been used on Eagle yet). Depending on the number of workers you want to run experiments, you can edit the file and change the following variable in up to as many core you want (default value is set to 2):
```shell
num_workers=2
```

After setting the number of workers, run the following

```
source run_rollout_workers.sh
```

## Rollout.py

The ```rollout.py``` script can be executed either by itself, or by using the ```multiple_workers.sh```. In the beginning, function ```rollout_loop``` creates two directories for saved games and TF models (these directories can also be given using the ```parser``` in function ```main```, depending on how we want that to work)

After initializing the policy network, the outer loop starts. The latest saved network is loaded and the inner rollout loops start. In ```save_game``` function, the games are saved. Because multiple instances running the same number of games, they are saved in a name format denoting the worker they are coming from. For example, **game_00_1.pickle** is the first game played on the first worker.

As soon as the rollout loops finish, the policy network is compiled. Then, a number of the latest games is sampled and they are used to train the network, which then is saved. Again, because experiments are running on multiple workers, there could be many instances of saved models. Therefore when a new network has to be loaded, we choose the latest that was saved based on time. The saved model names contain a number that defines the worker where they were saved. For example, **model_1.h5** denotes the model saved from first worker.



### Notes for reconsideration:

- We'll want to generate an experiment id at runtime, and use this to name/create directories, and also possibly namespaces in whatever DB we end up using (esp for Yuma which is shared).
<<<<<<< Updated upstream
- Games are saved as pickled objects, tf.data might be necessary later - Peter can help on that, but probably the two things can co-exist. As long as you have a data generator then it seems a tf.data pipeline can be built around it. we could have one for pickled inputs (testing), and another for DBs (experiments)
- For now, the entire model is saved (not just the weights), probably it has to be revisited.
- I need to check whether ```network.load_weights``` actually loads the latest model.
=======
- Games are saved as pickled objects, tf.data might be necessary later - let's get @pstjohn input here, but i think the two things can co-exist ... as long as you have a data generator then it seems a tf.data pipeline can be built around it. we could have one for pickled inputs (testing), and another for DBs (experiments)
- For now, the entire model is saved (not just the weights), probably it has to be revisited.
- I need to check whether ```network.load_weights``` actually loads the latest model.
>>>>>>> Stashed changes
