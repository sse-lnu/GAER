# STATUS

Repository: https://github.com/sse-lnu/GAER  
DOI: https://doi.org/10.5281/zenodo.xxxxxxx

We request the following ACM artifact badges:

1. Artifact Available
2. Artifact Functional
3. Artifact Reusable

## 1. Artifact Available

The artifact repository is publicly accessible and archived at Zenodo with an assigned DOI. It uses open licenses (BSD-3-Clause for source code and CC0 for data).

## 2. Artifact Functional

The artifact can be used to reproduce all the results provided in the paper. It uses standard methods for packaging and offers two ways to build and run it (pyproject.toml and Dockerfile). The various options, as well as instructions for reproducing the claims of the paper, are documented in the `README.md` file. The `README.md` also contains information about the formats of the included datasets and the output files.

The artifact is tested on macOS and Linux, with and without CUDA.

## 3. Artifact Reusable

The artifact implements two methods, our GAER and a reproduction of NEGAR, and also provides an easy way to run SARIF, another method used for comparison. It uses documented data formats, making it straightforward to extend the artifact with additional datasets or methods. The implementation of the pipeline is modular and easy to extend with new methods, such as alternative clustering algorithms. We provide implementations of the most common architecture recovery metrics in Python, which can be reused or extended.

The artifact supports several parameters that can be used to control the clustering search space, execute specific parts of the pipeline, and run selected datasets. It can output cluster assignments to make it easy to further investigate differences between the various methods and settings.

The artifact and the preprocessed data are distributed under open licenses to make it possible for other researchers to use and extend them.