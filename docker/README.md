# Docker

To make it as easy as possible to run GAER, NEGAR and SARIF, we include a set of Dockerfiles. To use them, make sure you have `docker` (or `podman`) installed. To build, use `docker build . -t [some:tag]` to build the image. All the images are configured to run the correct command, so you only need to supply the required parameters.

## GAER (CUDA)

The GAER and NEGAR Dockerfile requires CUDA and an NVIDIA GPU to run. Please ensure that you have, e.g., `docker` configured to use the NVIDIA toolkit so you can access the GPU from the container. You need to supply a directory for the results using the `-v` option. All the datasets are available in the image.

To run it, use 

```bash

docker run -it --rm --gpus all -v ./results:/gaer/results [some:tag]
```

This will run NEGAR and GAER-GAT on all datasets and save the results to `results/` in your pwd. If you want to change the parameters, you can pass any of the allowed parameters after the `[some:tag]`. You can, e.g., use

```bash

docker run -it --rm --gpus all -v ./results:/gaer/results [some:tag] --pipeline gaer --encoder gcn
```

to run GAER-GCN on all datasets.

For more information about which parameters are supporter, see `run_experiments.py` in `src/` or pass `--help` to the `docker run ...`

## SARIF

The SARIF Dockerfile should run on any x86 system. The image contains everything from the SARIF repository, including a demo dataset. To run it on other datasets, you need to supply the source code and ground truth in the correct format. 

To run SARIF on the demo dataset, distributed camera (HDC), use 

```bash 

docker run -it --rm -v ./results/:/sarif/Demo/results [some:tag] distributed_camera --gt distributed_camera_gt.json -o results
```

If you want to run it on other datasets, put these in a local folder and and pass this as another `-v` and adjust the parameters. For example, if you have chromium in `./data`, you could run 

```bash

docker run -it --rm -v ./data:/sarif/Demo/data -v ./results:/sarif/Demo/results [some:tag] data/chromium --gt data/chromium_gt.json -o results
```

We used the following sources to find the source code. The following were found via the SARIF repository. 

- Bash-4.2: [https://ftp.gnu.org/gnu/bash/bash-4.2.tar.gz](https://ftp.gnu.org/gnu/bash/bash-4.2.tar.gz)
- ArchStudio4: [https://github.com/isr-uci-edu/ArchStudio4](https://github.com/isr-uci-edu/ArchStudio4)
- Distributed Camera: [https://gitee.com/openharmony/distributed_camera](https://gitee.com/openharmony/distributed_camera) (Commit 46ff87)
- Drivers Framework: [https://gitee.com/openharmony/drivers_framework](https://gitee.com/openharmony/drivers_framework) (Commit 0e196f)
- OODT-0.2: [https://github.com/apache/oodt](https://github.com/apache/oodt) (Commit e927bc)
- Hadoop-0.19.0: [https://github.com/apache/hadoop](https://github.com/apache/hadoop) (Commit f9ca84)

In addition, we rely on Chromium from the Ubuntu source code package

- Chromium: 23.0.1271 / svn-171054 [https://launchpad.net/ubuntu/+source/chromium-browser/23.0.1271.97-0ubuntu0.12.04.1](https://launchpad.net/ubuntu/+source/chromium-browser/23.0.1271.97-0ubuntu0.12.04.1)
And Jabref and Teammates from the SAEroCon Repo:

- Jabref 3.7: [https://github.com/sebastianherold/SAEroConRepo/tree/master/systems/JabRef](https://github.com/sebastianherold/SAEroConRepo/tree/master/systems/JabRef)
- Teammates 5.110.0001: [https://github.com/sebastianherold/SAEroConRepo/tree/master/systems/TEAMMATES](https://github.com/sebastianherold/SAEroConRepo/tree/master/systems/TEAMMATES)

The ground truths are collected in the `GT/` directory. These are mainly collected from the SARIF repository. We converted a Chromium ground truth we found via the Internet Archive's Wayback Machine from RSF to json and created the json files for Jabref and Teammates from data in the SAEroCon repository.
