# GAER: Graph Auto-Encoders for Unsupervised Software Architecture Recovery

## Project Overview

**GAER (Graph Auto-Encoders for Unsupervised Software Architecture Recovery)** is a research prototype that automatically reconstructs the modular architecture of software systems from their source code. The artifact accompanies the paper:

> *GAER: Graph Auto-Encoders for Unsupervised Software Architecture Recovery*,  
> accepted at the ACM International Conference on the Foundations of Software Engineering (FSE) 2026.

A permanent version of this artifact is archived at:

DOI: https://doi.org/10.5281/zenodo.xxxxxxx

If you use this artifact in your research, please cite the associated FSE 2026 paper.
Citation information is provided in the [`CITATION.cff`](./CITATION.cff) file, which can be accessed via GitHub’s **“Cite this repository”** feature.

### Motivation

Software architecture documentation is often incomplete or becomes outdated as systems evolve. Recovering architectural modules manually is time-consuming and does not scale to large codebases. GAER addresses this challenge by providing a data-driven and automated approach to infer architectural decompositions directly from source code.

### Approach

GAER models a software system as a **heterogeneous, multi-relational dependency graph**, where:
- **Nodes** represent software entities (e.g., files or classes).
- **Edges** represent typed static dependencies (e.g., calls, inheritance, imports).
- **Node features** capture both **semantic information** (from code identifiers) and **directory structure** (from file paths).

A **Graph Auto-Encoder (GAE)** with a heterogeneous Graph Neural Network encoder learns architecture-aware embeddings of software entities. These embeddings are then clustered to recover architectural modules. The repository includes two encoder variants:
- **GAER-GCN** – based on Graph Convolutional Networks.
- **GAER-GAT** – based on Graph Attention Networks, typically providing stronger performance.

### Repository Contents

This repository provides:
- The **GAER** pipeline for architecture recovery.
- The **NEGAR** baseline pipeline for comparison.
- Scripts to **run experiments** on multiple open-source systems.
- **Processed datasets** and instructions for obtaining large files via Git LFS.
- Tools to compute **standard architecture recovery metrics**, including:
  - MoJoFM
  - Architecture-to-Architecture (A2A)
  - Cluster-to-Cluster Coverage (C2Ccvg)
  - Normalized TurboMQ

## Relation to the FSE 2026 Paper

The artifact supports the following claims:

- **C1 – Effectiveness:** GAER achieves competitive or superior alignment with ground-truth architectures compared to established baselines (e.g., SARIF and NEGAR) using standard metrics (MoJoFM, A2A, C2Ccvg, ARI).
- **C2 – Multi-Source Integration:** Integrating structural dependencies, directory hierarchy, and code semantics improves architecture recovery.
- **C3 – Encoder Effectiveness:** The GAT encoder generally outperforms the GCN variant.
- **C4 – Granularity Sensitivity:** The number of clusters significantly influences recovery quality.
- **C5 – Stability:** GAER demonstrates strong run-to-run stability across datasets.
- **C6 – Scalability:** GAER scales effectively to large software systems.

## Setup

The artifact is written in Python and has two main dependencies, [PyTorch](https://pytorch.org/) and [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/). We support two ways to manage the dependencies and run the artifact, `pyproject.toml` and Docker. We recommend using the `pyproject.toml` approach, but if you already use Docker and the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) it might be more convenient to use the Docker image.

Note that PyG supports NVIDIA CUDA or CPU, so if you are using Linux or Windows and have access to an NVIDIA GPU, you should use the default CUDA version for better performance. If you use macOS or do not have access to an NVIDIA GPU, you can use the CPU version.

We have tested the artifact on macOS (CPU) and Linux (CPU and GPU).

### `pyproject.toml`

We recommend that you use [uv](https://github.com/astral-sh/uv) to manage the dependencies. You can use `uv run` to automatically download the dependencies, create a temporary virtual environment, and run the program. 

```bash
uv run src/run_experiments.py --pipeline gaer --datasets Bash
```

The provided `pyproject.toml` assumes CUDA. If you want to use the CPU versions of the dependencies, you need to edit the file. We provide comments on where and what to change.

### Docker

Instructions for how to build and run the docker containers are available in `docker/`. Note that we only support Intel x86, so you need to build the images on Intel machine or use emulation. We also only support the CUDA version of the dependencies in the Docker image.

### Git LFS

Some of the datasets are large, so Git Large File Storage is required to check out the datasets. If you do not already have git-lfs installed, please follow the instructions at [https://git-lfs.com/](https://git-lfs.com/). Note that the docker images manage lfs during build, so if you prefer to run via docker, there is no need to install git-lfs.

## Quick Start (Sanity Check)

Once you have cloned the repo and installed the required tools (`uv` or Docker), you can use the following commands to ensure that you can run the program. The command runs our approach, GAER, on the Bash dataset. It can take some time to download all the dependencies, but once those are downloaded, the program should finish in seconds (allow up to a minute).

`uv`:

```bash
uv run src/run_experiments.py --pipeline gaer --datasets Bash
```

Docker:

```bash
docker run -it --rm --gpus all [name:tag] --pipeline gaer --datasets Bash
```

If `uv` does not work, make sure that you have the tool correctly set up. If it still does not work, check that git-lfs is enabled and the datasets in `data/` downloaded properly. If Docker does not work, make sure it is properly installed and can access the GPU.

## Running Experiments

We use `src/run_experiments.py` to start all the experiments. It supports a number of parameters to control which methods to run, which datasets to run them on, where the data is located, where the results are saved, and so on. We cover the most important ones below, please use the parameter `--help` to get the full list (or check the source code).

Use `--pipeline` to run `gaer`, `negar`, or `both`. You can control which encoder is used for GAER with `--encoder` set to `gat` or `gcn`.

You control which datasets to run with `--datasets` and either `all` to run all datasets or a combination of the following to run specific datasets.
- `AS4`
- `Bash`
- `Chrome`
- `Hadoop`
- `HDF`
- `HDC`
- `OODT`
- `Jabref`
- `TeamMates`
- `Libxml`

If the datasets are not available in `data/`, use `--data_dir` and the correct path. You can use `--out_dir` to specify where the results are stored. The default is `results/`

You can use `--no_eval` to skip computing the metrics (to speed up execution) and `--save_labels` to also save the clustering.

## Input and Output

We provide preprocessed datasets in csv format. Each dataset consists of two files, `dataset.csv` and `dataset_deps.csv`. The first contains information about the entities and the second about the dependencies. The preprocessed datasets were created using Depends v0.9.7. For more information about the source code used, see `docker/` and for more information about the ground truths, see `GT/`. 

For each entity, we store the following:
- `File`: the source filename, including path. 
- `ID`: a unique integer ID for the entity.
- `Member_Name`: the name of a member (e.g., method) of the entity.
- `Member_Type`: the type of the member, e.g., function.
- `Entity`: the entity name, fully qualified if applicable. E.g., the fully qualified Java class name.
- `Member_ID`: a unique integer ID for the member.
- `Module`: which (architectural) module the entity belongs to (based on the ground truth).
- `Module_List`: Some ground truth map a single entity to multiple modules. If this is the case, this contains the full list and `Module` is set to the module we decided to use (based on code inspection).
- `Duplicated`: Set to true if the ground truth contains multiple mappings for this entity.

For each dependency, we store the following. Note that many fields refer to entities in the previous file.
- `Source_ID`: reference to an entity ID 
- `Target_ID`: reference to an entity ID   
- `Source_File`: the source filename, including path.  
- `Target_File`:  the target filename, including path. 
- `Source_Member`: the name of the source.
- `Target_Member`: the name of the member. 
- `Source_Member_Type`: the type of the source member.
- `Target_Member_Type`: the type of the target member
- `Dependency_Type`: the type of the dependency, e.g., (method) Call.
- `Dependency_Count`: the number of such dependencies between the source and the target.
- `Is_Member_Level`: true of the dependency is between members.
- `Source_Member_ID`: reference to the member ID.
- `Target_Member_ID`: reference to the member ID.
- `Source_Modules`: similar structure as the previous file. A list that contains all possible ground truth mappings for the source.
- `Target_Modules`: all possible ground truth mappings for the target.
- `Source_Module`: the selected mapping for the source. 
- `Target_Module`: the selected mapping for the target.

The result is stored in a csv file with the following fields. Note that the files are named using the following pattern: `results_pipeline_date_time.csv`, e.g., `results_both_20260410_142229.csv`.

- `Dataset`: The dataset used
- `Clustering_Algorithm`: The clustering algorithm used
- `Recovered_clusters`: The K-value used for clustering (target number of clusters)
- `GT_clusters`: The number of clusters in ground truth architecture
- `MoJoFM`: the metric value (not included if `--no_eval`)
- `A2A`:  the metric value (not included if `--no_eval`)
- `C2CCvg_10`:  the C2C metric value under coverage threshold 0.10 (not included if `--no_eval`)
- `C2CCvg_33`:  the C2C metric value under coverage threshold 0.33 (not included if `--no_eval`)
- `C2CCvg_50`:  the C2C metric value under coverage threshold 0.50 (not included if `--no_eval`)
- `ARI`:  the metric value (not included if `--no_eval`)
- `TurboMQ_norm`: the metric value (not included if `--no_eval`)
- `Pipeline`: Which pipeline (`GAER`, `NEGAR`)
- `Encoder`: Which encoder (`gat`, `gcn`). Empty for `NEGAR`
- `GAE_loss`: The graph autoencoder loss
- `build_data_time(sec)`: The time (in seconds) to build the graph
- `Train_time(sec)`: ... to train
- `Cluster_time(sec)`: ... to cluster
- `Total_time(sec)`: The total time 

If `--save_labels` is provided, a cluster assignment is saved in a labels file. This follows the same naming pattern as the results file. This file uses the JSON format and contains a key per pipeline and dataset, e.g., `NEGAR::Bash`. The key holds an object that contains `labels`, which is an array of labels for each entity, and `k_used`, which is the K used for clustering. Note that the labels are in the range `0...k_used-1`. 

## Testing the claims

### C1 – Effectiveness

GAER achieves competitive or superior alignment with ground-truth architectures compared to established baselines (e.g., SARIF and NEGAR) using standard metrics (MoJoFM, A2A, C2Ccvg, ARI).

This can be verified by running:

```bash
uv run src/run_experiments.py --pipeline both --datasets all
```

The output contains values for all the standard metrics for GAER and NEGAR. The csv file can easily be loaded in, e.g., pandas (included as dependency) and analysed. 

To compare with SARIF, you can either use the values provided in our paper, the SARIF paper, or rerun the datasets using the provided Docker image of SARIF. Note that this requires a ground truth (available in `GT/`) and the source code (instructions on where to find it included in `docker/README.md`). 

### C2 – Multi-Source Integration

Integrating structural dependencies, directory hierarchy, and code semantics improves architecture recovery.

This can be verified by comparing against NEGAR, using the data generated in C1, which does not include some of these sources.

### C3 – Encoder Effectiveness

This can also be verified using the data generated in C1, but instead comparing GCN and GAT.

### C4 – Granularity Sensitivity

The number of clusters significantly influences recovery quality.

This can be verifed by using the `--k_min` and `--k_max` parameters to limit the search range or using `--user_k` to force a specific number of clusters.

Run GAER on Bash and pick the best cluster size in the range 5 to 10 clusters:

```bash
uv run src/run_experiments.py --pipeline gaer --datasets Bash --k_min 5 --k_max 10
```

Run GAER on Bash and use exactly 7 clusters:

```bash
uv run src/run_experiments.py --pipeline gaer --datasets Bash --user_k 7
```

### C5 – Stability

GAER demonstrates strong run-to-run stability across datasets.

This claim can be verified by running GAER multiple times and, e.g., producing a boxplot results. To reproduce Figure 2, use `--k_min` and `--k_max` for the top and `--user_k` for the bottom.

```bash
for i in {1..100}; do
    uv run src/run_experiments.py --pipeline both --datasets all --out_dir varicheck
done
```

### C6 – Scalability:

GAER scales effectively to large software systems

This can be verified by running the GAER pipeline a few times and analyzing the total time it takes. Since Chromium (`Chrome`) is the largest dataset, it is sufficient to run on it to see the time it takes. It can be helpful to use a small dataset, e.g., `Bash` to have a baseline.

```bash
for i in {1..5}; do
    uv run src/run_experiments.py --pipeline gaer --datasets Chrome --out_dir scalacheck
done
```

## License

### Code

This project’s source code is licensed under the BSD 3-Clause License.
See `LICENSE` for details.

### Data

All data in the `data/` directory is dedicated to the public domain under CC0 1.0.
See `LICENSE-DATA` for details.
