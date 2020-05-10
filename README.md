# graph_neural_net

## Project structure

```bash
.
├── loaders
|   └── dataset selector
|   └── data_generator.py # generating random graphs
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
|   └── test_maskedtensor.py # tests for maskedtensor.py
├── commander.py # tbc main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # tbc pipelines for training, validation and testing
```

### Install
Run
```
pip install -r requirements.txt
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

To save logs to [Neptune](https://neptune.ai/), you need to provide your own API key via the ```NEPTUNE_API_KEY``` environment variable.
