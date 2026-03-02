FROM alpine:latest AS build

RUN apk fix && apk --no-cache --update add git git-lfs

WORKDIR /git
RUN git lfs install
RUN git clone https://github.com/doubleblindcode55/GAER.git

FROM python:3.11.14-trixie
RUN pip install --no-cache networkx==3.2.1 gensim==4.3.3 nltk==3.8.1 tqdm==4.66.5 pandas==2.2.3 node2vec==0.5.0 scikit-learn==1.4.1.post1 && \
    pip install --no-cache pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --find-links https://data.pyg.org/whl/torch-2.8.0+cu129.html && \
    pip install --no-cache torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache torch_geometric

COPY --from=build /git/GAER /gaer
WORKDIR /gaer/src
ENTRYPOINT ["uv", "run", "run_experiments.py", "--data_dir", "/gaer/data", "--out_dir", "/gaer/results"]
