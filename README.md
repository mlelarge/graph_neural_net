# Graph neural networks and planted problems

## Overview
### Project structure

```bash
.
├── cpp_code # C++ code for exact solving to be compiled
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
|   └── data_handler.py # class handling the io of data and task-planning
|   └── helper.py  # base class for helping the selection of experiments when training a model
|   └── logger.py  # keeping track of most results during training
|   └── losses.py  # computing losses
|   └── maskedtensor.py # Tensor-like class to handle batches of graphs of different sizes
|   └── metrics.py # computing scores
|   └── mcp_solver.py # class handling the multi-threaded exact solving of MCP problems
|   └── minb_solver.py # class handling the multi-threaded exact solving of Min Bisection problems
|   └── optimizer.py # optimizers
|   └── searches.py # contains beam searches and exact solving functions
|   └── utility.py
|   └── vision.py # functions for visualization
├── article_commander.py # main file for computing the data needed for the figures
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # pipelines for training and validation
├── eval.py # testing models

```

### Dependencies
Dependencies are listed in `requirements.txt`. To install, run
```
pip install -r requirements.txt
```
For the TSP problem, it is required to install [pyconcorde](https://github.com/jvkersch/pyconcorde).

It is also needed to compile the `cpp_code/mcp_solver.cpp` (it uses C++ standard 14).
```
g++ -o "mcp_solver.exe" cpp_code/mcp_solver.cpp
```

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

## Data generation for planted against NP problems

All the parameters are stored in the `article_config.yaml` file. The experiments presented in the article are named `hhc`, `mcp` and `sbm`. To generate the data for the different problems, the `experiment` field must be changed to one of these values. Then just run :
```
python3 article_commander.py
```
The data will be generated in a subfolder `exps/[experiment_name]`.
