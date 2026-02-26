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

We provide two ways to manage the required dependencies, via docker or pyproject.toml. Instructions for how to build and run the docker containers are available in `docker/`.

We recommend that you use [uv](https://github.com/astral-sh/uv) if you want to manage the depdencies with the pyproject.toml. Note that we use NVIDIA CUDA by default. If you do not have access to an NVIDIA GPU, update the pyproject.toml to install CPU-versions of PyG and PyTorch.

Here follows some examples on how you can run GAER and NEGAR. Note that these exampels work simiarly if you run via docker.

Run GAER on a subset of the data with Default parameters

```bash

uv run src/run_experiments.py --pipeline gaer --datasets AS4 Hadoop

```
Run NEGAR baseline on a subset of the data with Default parameters

```bash
uv run src/run_experiments.py --pipeline negar --datasets Bash

```
Run all experiments with Default parameters and save recovered clusters Labels without evaluation

```bash

uv run src/run_experiments.py --pipeline both --datasets all --no_eval --save_labels

```
List available options

```bash

uv run src/run_experiments.py --help

```


## Outputs


The experiment runner produces:

- Cluster assignments (predicted module/cluster for each entity)

- Evaluation metrics (e.g., MoJoFM, A2A, C2C coverage, ARI; plus timing fields)

Outputs are written to the output directory (default: `results/`, or set via `--out_dir`).


