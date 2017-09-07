This project contains a simple API for object detection, able to deal with ensembles, and taking care of variables names, so that many similar models (same or similar architecture) can be ran all together at once on a single server. It is not graphic, and won't be either in the future versions, because it is supposed to run in background, and to be called by the platform. Basically the same concept as tensorflow serving, simpler, more flexible, but less secure for production use. It is used by [FiNC](https://finc.com/)'s AI team.

## Installation
Tensorflow is required, see [Installing TensorFlow](https://www.tensorflow.org/install) for instructions on how to install their release binaries or how to build from source.
The API also has the following requirements:
```shell
$ pip install easydict
$ pip install falcon
$ pip install falcon_multipart
$ pip install wsgiref
```

## Run the API
First, you need to clone my fork of the Object Detection model ([ODAPI](https://github.com/pierre-ecarlat/models/tree/master), and don't forget to follow the installation procedures (compile the protobuf and export the new $PYTHONPATH). This dependency might be removed (focused) in the future.  
If you want to add a model, go on the `api.py` script, and simply add it to the appropriated list (`ensembles_list` or `single_models_list`). Dependencies:
- Ensemble: each model in the ensemble require a `configs/[model.name].config` and the `models/[model.name]/checkpoint_files`
- Single model: the model require a `models/[model.name].pb` saved model for inference


Then, run the API in a permanent terminal (in a `screen` for example):
```shell
# Might take some time if many models need to be loaded, will list 
# the successfully served models when it'll be done
$ python api.py
```
Then,
```shell
$ python request.py --image path/to/image.jpg --model model_name
```
