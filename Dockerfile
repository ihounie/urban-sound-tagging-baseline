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
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing &&     apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion &&     apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh &&     /bin/bash ~/anaconda.sh -b -p /opt/conda &&     rm ~/anaconda.sh &&     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&     echo "conda activate base" >> ~/.bashrc &&     find /opt/conda/ -follow -type f -name '*.a' -delete &&     find /opt/conda/ -follow -type f -name '*.js.map' -delete &&     /opt/conda/bin/conda clean --yes -afy
ENV PATH=/opt/conda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN conda update -y conda
RUN conda update --all
RUN apt install --yes gcc
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
RUN yes | pip3 install tensorflow-gpu==1.13.1
# (Optional) install Jupiter notebooks to run examples
RUN yes | pip3 install notebook

ENTRYPOINT [ "/bin/bash"]
CMD [ "/bin/bash" ]
