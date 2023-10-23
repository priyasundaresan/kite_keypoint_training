#FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM nvidia/cudagl:11.2.0-base-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN rm /etc/apt/sources.list.d/*
RUN apt-get clean 

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libhdf5-dev \
      libglib2.0-0 \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim wget \
    build-essential cmake \
    libopenblas-dev

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip

RUN ln -sfn /usr/bin/python3.8 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

#RUN pip install --upgrade pip && \
#    pip install \
#        torch==1.12.1 \
#        torchvision \
#        opencv_python \
#        numpy \
#        Cython \
#        matplotlib 

RUN pip install --upgrade pip && \
    pip install \
        torch==1.7.0 \
        torchvision==0.8.0 \
        opencv_python \
        numpy \
        Cython \
        matplotlib \
        transformers

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install git+https://github.com/openai/CLIP.git \
        openai

WORKDIR /host
CMD ["/bin/bash"]
