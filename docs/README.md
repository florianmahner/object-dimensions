# Object Dimensions in Humans :elf: and DNNs :spider_web:

Before using this repository, we recommend to install all packages. We provide a `makefile` for this:

```bash
make install
```

Finally activate the environment and install the package locally

```bash
conda activate deep
pip install -e .
```


## Content of this repository

We learn interpretable object embeddings using a triplet task. We therefore need to either extract triplets or use human generated ones.

## Example Data

We provide an example script to download all the things and things+ data. Execute it as 

```bash
bash scripts/get_things_data.sh
```

## Extracting Triplets

We can pass arguments to the all python scripts using a `config.toml` file. A couple of example config files are provided in `./configs`. To extract triplets from DNN representations we can for example call:

```bash
python tripletize.py --config "./configs/tripletize.toml"
```

## Training the model

We can train the model using a variety of different hyperparameters and optimizations. 
For a list of arguments execute:

```bash
python main.py --help
```

Most importantly, we can train the model determinstically using MLE as introduced with [SPoSE](https://www.nature.com/articles/s41562-020-00951-3) or variationally as in [VICE](https://arxiv.org/abs/2205.00756). We enable these through the flags `--method {deterministic, variational}`.

To simplify recreating the experiments, we have summarized example configurations in `.toml` files. The arguments in the file are passed onto the argument parser and can be used for training (see [here](https://github.com/florianmahner/toml-argparse) for the corresponding repo). We can execute a model run using the configuration files as follows:


```bash
python main.py --config "./configs/train_behavior.toml" --method "deterministic"
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