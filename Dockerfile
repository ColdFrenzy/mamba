# Download GPU-enabled pytorch image
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# Download base image ubuntu 
# FROM ubuntu:20.04
# Download ubuntu with nvidia cuda
# FROM nvidia/cuda:11.7.0-base-ubuntu20.04
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

LABEL description="Custom Docker image to run the Dreamer Multi-Agent Architecture"

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

# changed python37-distutils to python37-distutils-extra
RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        tmux sudo \
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
        apt-get install -y python3.8 && \
        update-alternatives --install /usr/bin/python3 python /usr/bin/python3.8 1 && \
        apt-get install -y python3.8-distutils && \ 
        wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash robot && echo "robot:robot" | chpasswd
RUN usermod -u $UID robot && groupmod -g $GID robot

RUN adduser robot sudo  && \
    adduser robot audio  && \
    adduser robot video  && \
    adduser robot dialout
    
# create a .dockerignore file to exclude some file
COPY . ./mamba

RUN cd ./mamba && pip install --upgrade pip setuptools && pip install -e .\
        git+https://github.com/oxwhirl/smac.git
	

CMD ['bin/bash']