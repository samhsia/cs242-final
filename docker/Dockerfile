# FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM hubris.int.seas.harvard.edu/harvardacc/clusterbase:gpu

RUN apt-get update && apt-get install -y \
  build-essential \ 
  emacs \
  git \
  libffi-dev \
  linux-tools-4.4.0-171-generic \
  numactl \
  python3 \
  python3-dev \
  python3-pip \
  tmux

# Upgrade pip (already installed)
RUN pip3 install --upgrade "pip < 21.0"

RUN pip3 install \
  future \
  matplotlib \ 
  notebook \
  numpy \
  scikit-learn \
  torch \
  torchvision==0.5 \
  tqdm
  
WORKDIR /home/schsia/cs242-final