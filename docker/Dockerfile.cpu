FROM ubuntu:16.04

# Python3 and native deps for bindings (pyosmium, rtree).
#
# libsm6 native dep. is required for opencv-contrib-python.
# See: https://github.com/skvark/opencv-python/issues/90
RUN apt-get update -qq && \
    apt-get install -qq -y -o quiet=1 \
    python3 python3-dev python3-tk python3-pip build-essential libboost-python-dev libexpat1-dev zlib1g-dev libbz2-dev libspatialindex-dev libsm6

WORKDIR /app
ADD . /app

ARG TCH=0.4.0
ARG CU=91
ARG PY=35

RUN python3 -m pip install http://download.pytorch.org/whl/cpu/torch-${TCH}-cp${PY}-cp${PY}m-linux_x86_64.whl     && \
    python3 -m pip install -r deps/requirements-lock.txt

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
