# GAER: Graph Auto-Encoders for Unsupervised Software Architecture Recovery



This repository contains code to run two software architecture recovery pipelines:

- gaer: graph autoencoder-based architecture recovery

- negar: baseline pipeline

- both: runs both pipelines



## Setup



```bash

git clone https://github.com/doubleblindcode55/GAER.git

cd GAER

```

### Running Experiments



Example run GAER on a subset of the data with Default parameters

```bash

python run\_experiments.py --pipeline gaer --datasets AS4 Hadoop

```
Example run NEGAR baseline on a subset of the data with Default parameters

```bash
python run\_experiments.py --pipeline negar --datasets Bash

```
Run all experiments with Default parameters and save recovered clusters Labels without evaluation

```bash

python run_experiments.py --pipeline both --datasets all --no_eval --save_labels

```
List available options

```bash

python run\_experiments.py -h

```


## Outputs


The experiment runner produces:

- Cluster assignments (predicted module/cluster for each entity)

- Evaluation metrics (e.g., MoJoFM, A2A, C2C coverage, ARI; plus timing fields)

Outputs are written to the output directory (default: `results/`, or set via `--out\_dir`).


