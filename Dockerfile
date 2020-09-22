# Use a docker image as base image
FROM nvidia/cuda:10.0-cudnn7-runtime
# Declare some ARGuments
ARG PYTHON_VERSION=3.6
# Installation of some libraries / RUN some commands on the base image
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN apt-get install -y  libsndfile1 libsndfile1-dev
### audio converters
RUN apt update && apt-get install -y ffmpeg mpg123
### sox package to adjust sample rate.
RUN apt-get install -y libsox-fmt-all libsox-dev sox

RUN apt-get install -y tmux
RUN pip3 install jupyter notebook
RUN conda install -c anaconda llvm
RUN conda install -c numba llvmlite
RUN pip3 install sed_eval
RUN cd ~/ && git clone https://github.com/ihounie/urban-sound-tagging-baseline.git
RUN cd ~/urban-sound-tagging-baseline && ./setup.sh
RUN pip3 install comet_ml
RUN apt-get install -y vim
RUN apt-get install -y libsndfile1 libsndfile1-dev libsox-fmt-all libsox-dev sox
# Install Tensorflow with GPU support
RUN pip3 install --yes tensorflow-gpu==1.13.1
# (Optional) install Jupiter notebooks to run examples
RUN pip3 install --yes notebook

ENTRYPOINT [ "/bin/bash"]
CMD [ "/bin/bash" ]
