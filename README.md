# Graph neural networks for the Quadratic Assignment Problem

## Overview
### Project structure

```bash
.
├── loaders
|   └── dataset selector
|   └── data_generator.py # generating random graphs
|   └── test_data_generator.py
|   └── siamese_loader.py # loading pairs 
├── models
|   └── architecture selector
|   └── layers.py # equivariant block
|   └── base_model.py # powerful GNN Graph -> Graph
|   └── siamese_net.py # GNN to match graphs
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training
|   └── metrics.py # computing scores
|   └── losses.py  # computing losses
|   └── optimizer.py # optimizers
|   └── utility.py
|   └── maskedtensor.py # Tensor-like class to handle batches of graphs of different sizes
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # pipelines for training and validation
├── eval.py # testing models
```


## Dependencies
Dependencies are listed in `requirements.txt`. To install, run
```
pip install -r requirements.txt
```
## Training 
Run the main file ```commander.py``` with the command ```train```
```
python train commander.py
```
To change options, use [Sacred](https://github.com/IDSIA/sacred) command-line interface and see ```default.yaml``` for the configuration structure. For instance,
```
python commander.py train with cpu=No data.generative_model=Regular train.epoch=10 
```
You can also copy ```default.yaml``` and modify the configuration parameters there. Loading the configuration in ```other.yaml``` (or ```other.json```) can be done with
```
python commander.py train with other.yaml
```
See [Sacred documentation](http://sacred.readthedocs.org/) for an exhaustive reference. 

To save logs to [Neptune](https://neptune.ai/), you need to provide your own API key via the dedicated environment variable.

The model is regularly saved in the folder `runs`.

## Evaluating

There are two ways of evaluating the models. If you juste ran the training with a configuration ```conf.yaml```, you can simply do,
```
python commander.py eval with conf.yaml
```
You can omit ```with conf.yaml``` if you are using the default configuartion.

If you downloaded a model with a config file from here, you can edit the section ```test_data``` of this config if you wish and then run,
```
python commander.py eval with /path/to/config model_path=/path/to/model.pth.tar
```
