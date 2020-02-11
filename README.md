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
|   └── logger.py  # tbd
|   └── metrics.py # tbd
|   └── plotter.py # tbd
|   └── losses.py  # compute losses
|   └── optimizer.py # optimizers
├── commander.py # tbd main file from the project serving for calling all necessary functions for training and testing
├── args.py # tbd parsing all command line arguments for experiments
├── trainer.py # tbc pipelines for training, validation and testing
```
