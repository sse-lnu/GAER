# Installation instructions

## Prerequisites

The project provides two ways to install the dependencies: a `pyproject.toml` file and a Dockerfile. We recommend that you use `uv` and the `pyproject.toml`. If you want to use the Dockerfile, you need an Intel x86 platform (or emulation) to build the image and an NVIDIA GPU to run the container.

If you want to use CUDA, we assume that you have an NVIDIA GPU and that the drivers are working. You can verify this with, e.g., `nvidia-smi`.

### Install `uv`

Please refer to [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/) for instructions on how to install uv. 

### Install Docker

Please refer to [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/) for instructions on how to install Docker. We have only tested the containers using Linux (with docker and podman), so if you want to try it on Windows, we recommend that you use WSL2. 

Install the NVIDIA Container Toolkit to allow containers to access your GPU. Please refer to [https://github.com/NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) for instructions.

### Install git large file support (git-lfs)

The datasets are large and require large file support. Please refer to [https://git-lfs.com](https://git-lfs.com) for instructions on how to install git-lfs. Once installed, make sure you enable it with `git lfs install`.

## Installing

Everything needed to run the artifact is available in the repository, so simply clone https://github.com/sse-lnu/GAER.git .

```bash
git clone https://github.com/sse-lnu/GAER.git
cd GAER
git lfs pull
```

If you want to build the Docker containers, instructions and Dockerfiles are available in the `docker/` directory. Note that we also provide a Dockerfile to reproduce the results of SARIF. This is not needed to run our artifact.

## Running

Note that the default configuration assumes the use of an NVIDIA GPU with CUDA support.  If a GPU is not available, the artifact can still be executed using CPU-only versions of PyTorch and PyTorch Geometric. To do so, edit the `pyproject.toml` file and replace the CUDA-specific dependencies with the CPU versions as indicated by the comments in the file.

Use the following command to run our method, GAER, on the Bash dataset. Bash is small, so even running on a slow CPU, the script should finish in a few seconds (but note that the dependencies will be installed the first time it is run, so that will add some extra time).  When the `--no_eval` flag is used, the output CSV contains a reduced set of fields focused on runtime information. Running the experiments without this flag will produce additional evaluation metrics such as MoJoFM, A2A, C2Ccvg, ARI, and TurboMQ_norm, as described in the README.

```bash
uv run src/run_experiments.py --pipeline gaer --datasets Bash --no_eval
```

The script should produce a CSV file in the `results/` directory. The filename is printed to the console. Opening this file should reveal a single row of numeric values, confirming that the artifact is functioning correctly. The exact values may vary depending on the execution environment, but the structure should match the example below.


```csv
Dataset,Clustering_Algorithm,Recovered_clusters,Pipeline,Encoder,GAE_loss,build_data_time(sec),Train_time(sec),Cluster_time(sec),Total_time(sec)
Bash,AHC,16,GAER,gat,1.5752469301223755,0.7181,1.7719,0.1794,2.6694
```

Once you are sure it works, you try the following command to look at all available options.

```bash
uv run src/run_experiments.py --help
```

You can also run the following to run all methods (GAER and NEGAR) on all datasets. Note that this can take some time, especially if running on a slower computer or without a GPU. The largest dataset, Chrome, took about 10 minutes for GAER and about 100 minutes for NEGAR on a (quite slow) two-core cloud vps. The same dataset took about 2 minutes for GAER and about 6 minutes for NEGAR on a M1 Max Macbook Pro. On an H100 NVIDIA GPU, GAER takes about 40 seconds on the Chrome dataset. A high-end consumer GPU (RTX 4090 or RTX 5090) has similar performance (for GAER). Note that NEGAR does not benefit from GPU acceleration, so expect 5-10 minutes when running on a recent desktop/server CPU. 

If you do not have access to a GPU or a recent CPU, we recommend that you exclude Chrome. Running all datasets except Chrome takes about 15 minutes on the slow two-core cloud vps or about 1-2 minutes on the M1 Max Macbook Pro.

```bash
uv run src/run_experiments.py --pipeline both --datasets all
```

or, if you want to skip Chrome.

```bash
uv run src/run_experiments.py --pipeline both --datasets AS4 Bash Hadoop HDF HDC OODT Jabref TeamMates Libxml
```

For more information on the various options, the datasets and the output formats, please see the README.md file in the repository.
