FROM alpine:latest AS build

RUN apk fix && apk --no-cache --update add git git-lfs

WORKDIR /git
RUN git lfs install
RUN git clone https://github.com/anonymous2f4a9d/SARIF_FSE23

FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless libyaml-dev libjansson-dev libxml2-dev


COPY --from=build /git/SARIF_FSE23 /sarif
RUN chmod 755 /sarif/Demo/sarif_demo /sarif/Demo/ext_tools/ctags_linux/ctags
WORKDIR /sarif/Demo
ENTRYPOINT ["./sarif_demo"]
