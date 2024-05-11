# Download GPU-enabled pytorch image
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# Download base image ubuntu 
# FROM ubuntu:20.04
# Download ubuntu with nvidia cuda
# FROM nvidia/cuda:11.7.0-base-ubuntu20.04
FROM nvidia/cuda:11.7.0-base-ubuntu20.04

LABEL description="Custom Docker image to run the Dreamer Multi-Agent Architecture"

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        software-properties-common \
        libfreetype6-dev \
        pkg-config \
        curl \
        git \
        vim && \
        apt-get install glibc-source -y && \
        apt-get install -y wget && \
        apt-add-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y python3.7 && \
        update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 1 && \
        apt-get install -y python37-distutils && \
        wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# create a .dockerignore file to exclude some file
COPY . ./mamba

RUN cd ./mamba && pip install wheel flatland-2.2.2/ && pip install \
	numpy~=1.18.5 \
	torch~=1.7.0 \
	ray~=1.13.0 \
	git+https://github.com/oxwhirl/smac.git \
	wandb~=0.13.11 \
	argparse~=1.4.0 \

CMD ['bin/bash']
