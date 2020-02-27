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
├── commander.py # tbc main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # tbc pipelines for training, validation and testing
```

### Install
Run
```
pip install -r requirements.txt --pre -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```

