# Object Dimensions in Humans :elf: and DNNs :spider_web:


## Installation

Before using this repository, we recommend to install all packages using [poetry](https://python-poetry.org/). If you have not installed, please install it first and add it to your path

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

Then, clone the repository, navigate to the project root and install the dependencies using poetry
```bash
git clone git@github.com:florianmahner/object-dimensions.git
cd object-dimensions
poetry install
poetry shell
```

## Content of this repository

We learn interpretable representational embeddings from behavioral responses to natural images in a triplet odd-one-out task. There are in principle two way to obrain these choices:

1.  Running behavioral experiments in humans
2.  Simulating triplet choices from the similarity of any representation space. In this case we can sample these judgments from any representation space that allows for a similarity measure (ie DNN activations, neural recordings, etc.)


This repository allows you to both simulate behavioral choices and then use these simulated or actual behavioral choices to train an embedding model. We 



## Example Data

We use DNN and human representation in response to natural images. For our experiments, we use the THINGS dataset, which is a collection of 1,854 images of everyday objects. In principle, the approach can be applied to arbitrary datasets. In principle the approach can be applied to arbitrary datasets.
If you want to download the THINGS and THINGS+ data easily, we provide a makefile for this. Simply execute

```bash
make data
```

## Extracting Triplets

To train the model, we need to have behavioral responses to triplets. We can either simulate these responses from any representation space or use actual behavioral responses. If you want to use an representation to extract triplets, you can use the `run_tripletization.py` script. 

We can pass arguments to the all python scripts using a `config.toml` file. A couple of example config files are provided in `./configs`. To extract triplets from DNN representations we can for example call. See [tomlparse](https://github.com/florianmahner/tomlparse) for more details on how to use TOML files with the argparse module.

```bash
python run_tripletization.py --config "./configs/tripletize.toml"
```

## Training the model

Afterwards, we can train the model using a variety of different hyperparameters and optimizations. 
For a list of arguments execute:

```bash
python run_optimization.py --help
```

Most importantly, we can train the model determinstically using MLE as introduced with [SPoSE](https://www.nature.com/articles/s41562-020-00951-3) or variationally as in [VICE](https://arxiv.org/abs/2205.00756). We enable these through the flags `--method {deterministic, variational}`.

To simplify recreating the experiments, we have summarized example configurations in `.toml` files. The arguments in the file are passed onto the argument parser and can be used for training (see [here](https://github.com/florianmahner/toml-argparse) for the corresponding repo). We can execute a model run using the configuration files as follows:


```bash
python run_optimization.py --config "./configs/train_behavior.toml" --method "deterministic"
```


The model run is stored in different paths, depending on whether the modality are behavior or deep neural net triplets:

```
Behavior: 
    "./log_path/identifier/behavior/n_samples/prior/init_dim/batch_size/beta/seed"

DNN: 
    "./log_path/identifier/deep/model_name/module_name/n_samples/prior/init_dim/batch_size/beta/seed"
```

In each run path the following is stored

```
path
├── config.toml
├── training.log
├── params
├── └── parameters.npz
├── tboard
├── checkpoints
├── └── checkpoint_epoch_*.tar
```

## Evaluating the model

Our final model is stored in `parameters.npz` alongside all training configurations. For SPoSE we model only the mean of our embedding, which is saved, and for VICE the mean and variance of our Gaussian variational distribution.


## Main Experiments of the Paper
We provide the main experiments of the paper in the `./experiments` folder. See the README there for an overview.
