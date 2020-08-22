# Tensorflow-serving example code

An example of using tensorflow-serving to give model predictions for GNNs building nfp custom layers. This is looking pretty promising if we can get a GPU-enabled singularity image to work on Eagle. Specifically, this might be a good way to leverage heterogenous compute (GPUs and CPUs) by keeping the tensorflow models on GPU nodes (as served models) while running rollouts on CPUs.

Some additional links:
- [Using GPUs with docker](https://www.tensorflow.org/tfx/serving/docker#serving_with_docker_using_your_gpu)
- [Dynamically discover and serve new model versions](https://www.tensorflow.org/tfx/serving/serving_advanced)
- [Issues using multiple GPUs with serving](https://github.com/tensorflow/serving/issues/311)
