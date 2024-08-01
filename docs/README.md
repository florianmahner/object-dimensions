# Object Dimensions in Humans :elf: and DNNs :spider_web:

## Overview

This repository provides the code to reproduce the main parts of the paper . In addition, it provides the code to learn interpretable representational embeddings from behavioral responses to natural images using a triplet odd-one-out task.

This repository is split into two parts:
1. Code and experiments to reproduce the main results of the paper [Dimensions underlying the representational alignment of deep neural networks with humans](arxiv.org/pdf/2406.19087). See [Main Experiments of the Paper](#main-experiments-of-the-paper) for more details.
2. Code to learn interpretable representational embeddings from behavioral responses to natural images using a triplet odd-one-out task. See [Learning Representational Embeddings](#learning-representational-embeddings) for more details.


## Installation and System Requirements

### Step 1: Install Poetry

This project uses Python 3.9.12 and Poetry for dependency management. Most experiments can be run using a normal desktop computer in a reasonable amount of time. However, most
experiments require Pytorch and an NVIDIA GPU. 

First, install [Poetry](https://python-poetry.org/):

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Step 2: Clone the Repository

Clone the repository and navigate to the project root:

```bash
git clone git@github.com:florianmahner/object-dimensions.git
cd object-dimensions
```

### Step 3: Install Dependencies

Install the project dependencies using Poetry:

```bash
poetry install
poetry shell
```


## Main Experiments of the Paper 

Prior to running the experiments, download the required data from [figshare](https://figshare.com/articles/dataset/VGG16_embedding/26424238?file=48071587) and place them in the `./data` folder. are required for the experiments. We use the THINGS dataset, which consists of 1,854 images of everyday objects. We provide a script download the THINGS and THINGS+ data. To download the data, run:

```bash
make data
```

The code for all experiments is detailed in the [experiments](../experiments/README.md)
     folder. We provide config files for each experiment in the [configs](../configs) folder. 


## Learning representational embeddings

We learn interpretable representational embeddings from behavioral responses to natural images using a triplet odd-one-out task. The choices can be obtained by:

1. Running behavioral experiments with humans.
2. Simulating triplet choices from any representation space (e.g., DNN activations, neural recordings).

The repository supports both the simulation of behavioral choices and the use of actual behavioral data to train an embedding model.

Triplets can be simulated from any representation space or collected from actual behavioral responses. If you have actual behavioral reponses, make sure that the data is of shape [n_samples, 3], where each row contains the indices of the triplets and the *last column* by definition denotes the odd one out.

### Simulating Triplets

Use the `run_tripletization.py` script to simulate triplets. Example configurations are provided in the [configs](../configs)  folder.

To extract triplets from DNN representations, run:

```bash
python run_tripletization.py --config "./configs/tripletize.toml"
```

## Training the Model

Train the model with different hyperparameters and optimization methods. For a list of available arguments, run:

```bash
python run_optimization.py --help
```

### Training Methods

You can train the model deterministically using MLE (as in [SPoSE](https://www.nature.com/articles/s41562-020-00951-3)) or variationally (as in [VICE](https://arxiv.org/abs/2205.00756)) by specifying the `--method` flag.

Example command:

```bash
python run_optimization.py --config "./configs/train_behavior.toml" --method "deterministic"
```

### Model Run Paths

The model run paths are organized based on the modality:

- **Behavior:** `./log_path/identifier/behavior/n_samples/prior/init_dim/batch_size/beta/seed`
- **DNN:** `./log_path/identifier/deep/model_name/module_name/n_samples/prior/init_dim/batch_size/beta/seed`

Each run path includes the following files:

```
path
├── config.toml
├── training.log
├── params
│   └── parameters.npz
├── tboard
├── checkpoints
│   └── checkpoint_epoch_*.tar
```

## Evaluating the Model

The final model parameters are stored in `parameters.npz`, along with all training configurations. For SPoSE, only the mean of the embedding is modeled, while for VICE, both the mean and variance of the Gaussian variational distribution are modeled.


## Contact

For any questions or issues, please open an issue on GitHub or contact the repository maintainers.