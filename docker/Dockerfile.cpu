FROM ubuntu:18.04

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false" CFLAGS="-mavx2"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libspatialindex-c4v5 libglib2.0-0 \
    wget gcc yasm cmake make python3-dev zlib1g-dev libwebp-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.2.3 pip-tools==4.1.0 && \
    python3 -m piptools sync

RUN python3 -m pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl && \
    python3 -m pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

RUN python3 -c "from torchvision.models import resnet50; resnet50(True)"

RUN wget -q https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.3.tar.gz -O libjpeg-turbo.tar.gz && \
    echo "a69598bf079463b34d45ca7268462a18b6507fdaa62bb1dfd212f02041499b5d libjpeg-turbo.tar.gz" | sha256sum -c && \
    tar xf libjpeg-turbo.tar.gz && \
    rm libjpeg-turbo.tar.gz && \
    cd libjpeg-turbo* && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DREQUIRE_SIMD=On -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && \
    rm -rf libjpeg-turbo*

RUN python3 -m pip uninstall -y pillow && \
    python3 -m pip install --no-binary :all: --compile pillow-simd==6.0.0.post0

COPY . .

ENTRYPOINT ["/usr/src/app/rs"]
CMD ["-h"]
