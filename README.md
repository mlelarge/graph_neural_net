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

### Install

### Run
Run the main file ```commander.py```
```
python commander.py
```
To change options, use [Sacred](https://github.com/IDSIA/sacred) command-line interface and see ```default.yaml``` for the configuration structure. For instance,
```
python commander.py with cpu=No data.generative_model=Regular train.epoch=10 
```
You can also copy ```default.yaml``` and modify the configuration parameters there. Loading the configuration in ```other.yaml``` (or ```other.json```) can be done with
```
python commander.py with other.yaml
```
See [Sacred documentation](http://sacred.readthedocs.org/) for an exhaustive reference. 

To save logs to [Neptune](https://neptune.ai/), you need to provide your own API key via the dedicated environment variable.

## Dependencies
Dependencies are listed in `requirements.txt`. To install, run
```
pip install -r requirements.txt
```
## Training 
Running
```
python commander.py
```
will train a model with the parameters defined in `default.yaml`.

The model is regularly saved in the folder `runs`. More precisely, with the current parameters in `default.yaml`, it is saved in `runs/ER_std/QAP_ErdosRenyi_ErdosRenyi_50_1_0.05`.

## Evaluating

The script `eval.py` is used to test a model. With the example above,
```
python eval.py --name ER_std --model-path runs/ER_std/QAP_ErdosRenyi_ErdosRenyi_50_1_0.05
```
will retrieve the trained model and evaluated it on a test dataset. More options are available in `eval.py`.
