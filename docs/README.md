# Object Dimensions in Humans and DNNs 


Before using this repository, we recommend to install all packages inside the provided virtual environment:

```bash
conda env create -f environment.yaml
conda activate deep
```

## Content of this repository

We learn interpretable object embeddings using a triplet task. We therefore need to either extract triplets or use human generated ones.

## Extracting Triplets

We can pass arguments to the all python scripts using a `config.toml` file. A couple of example config files are provided in `/configs`. To extract triplets from DNN representations we can for example call:

```bash
python tripletize.py --config "configs/tripletize.toml"
```

## Training the model

We can train the model using MLE as introduced with SPoSE or variationally as in VICE.
To run the model for behavior for instance and using SPoSE please execute

bash
```
python main.py --configs "/configs/train_behavior.toml" --method "deterministic"
```


