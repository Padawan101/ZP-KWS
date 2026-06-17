FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install dependency
RUN apt-get update && apt-get install -y gnupg2
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv A4B469963BF863CC
RUN apt-get update -y && apt-get install -y \
    git \
    libsndfile1 

# Install python packages
RUN python -m pip install --upgrade pip && pip install \
    levenshtein \
    six \
    audioread \
    librosa  \
    PySoundFile \
    scipy \
    tqdm \
    pandas \
    nltk \
    inflect \
    torchmetrics \
    tensorboard \
    accelerate \
    lmdb \
    thop \
    scikit-learn \
    geatpy \
    matplotlib

RUN python -m pip uninstall -y numpy
RUN python -m pip install numpy==1.26.4
RUN python -m pip uninstall -y protobuf
RUN python -m pip uninstall -y google
RUN python -m pip install "protobuf<4.21.0"

WORKDIR /home
