# Core Object Dimensions of Deep Neural Networks and Humans

## Overview

This repository contains code to reproduce the key results of our paper. Additionally, it provides tools to learn interpretable representational embeddings from behavioral responses to natural images using a triplet odd-one-out task.

The repository is organized into two main sections:

1. **Reproducing the main experiments** (see [Main Experiments](#main-experiments))
2. **Learning interpretable representational embeddings** (see [Learning Representational Embeddings](#learning-representational-embeddings))

## Installation

### 1. Install Poetry

This project requires **Python 3.9.12** and **Poetry** for dependency management. Most experiments can be run on a standard desktop, but a **GPU** (with PyTorch) is recommended.

First, install [Poetry](https://python-poetry.org/):

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone the Repository

```bash
git clone git@github.com:florianmahner/object-dimensions.git
cd object-dimensions
```

### 3. Install Dependencies

```bash
poetry install
poetry shell
```

## Main Experiments

### Downloading Data

Before running experiments, download the required data from [OSF](https://osf.io/nva43/) by executing:

```bash
make data
```

Additionally, this project uses the **THINGS dataset** (1,854 images of everyday objects). To download THINGS and THINGS+, run:

```bash
make images
```

### Running Experiments

Each experiment is configured via a `.toml` file in the [`configs`](../configs) folder. A detailed guide is available in the [experiments README](../experiments/README.md).

## Learning Representational Embeddings

We learn interpretable embeddings from behavioral responses to natural images using a **triplet odd-one-out task**. Choices can come from:

1. **Human behavioral experiments**
2. **Simulated choices from representational spaces** (e.g., DNN activations, neural recordings)

The repository supports both **simulating triplets** and **using real behavioral data** to train an embedding model. A small demo is available in [`scripts/demo.ipynb`](../scripts/demo.ipynb).

> **Note:**  
> If using real behavioral data, ensure the format is `[n_samples, 3]`, where each row contains triplet indices, with the *last column* denoting the odd one out.

### Simulating Triplets

Use `run_tripletization.py` to generate triplets. Example configurations are in [`configs`](../configs).

To extract triplets from DNN representations:

```bash
python run_tripletization.py --config "./configs/tripletize.toml"
```

## Training the Model

To explore training options, run:

```bash
python run_optimization.py --help
```

### Training Methods

The model can be trained using:

- **Deterministic Maximum Likelibood Estimation** (as in [SPoSE](https://www.nature.com/articles/s41562-020-00951-3))
- **Variational Inference** (as in [VICE](https://arxiv.org/abs/2205.00756))

Example training command:

```bash
python run_optimization.py --config "./configs/train_behavior.toml" --method "deterministic"
```

### Model Run Paths

Training logs and results are organized as follows:

#### **For Behavioral Data**
```
./log_path/identifier/behavior/n_samples/prior/init_dim/batch_size/beta/seed
```

#### **For DNN Representations**
```
./log_path/identifier/deep/model_name/module_name/n_samples/prior/init_dim/batch_size/beta/seed
```

Each run contains:

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

The trained model parameters are saved in `parameters.npz`, along with the full training configuration.

- **SPoSE**: Models only the **mean** of the embedding.  
- **VICE**: Models both the **mean** and **variance** of the Gaussian variational distribution.

## Contact

For questions or issues, open a GitHub issue or contact the maintainers.
