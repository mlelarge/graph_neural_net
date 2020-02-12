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
|   └── logger.py  # tbc
|   └── metrics.py # tbc
|   └── plotter.py # tbd
|   └── losses.py  # compute losses
|   └── optimizer.py # optimizers
├── commander.py # tbc main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # tbc pipelines for training, validation and testing
```
