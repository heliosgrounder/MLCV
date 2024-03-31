FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    build-essential \ 
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/cache/apt/archives \
    && rm -rf /var/lib/apt/lists

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install build

COPY . /data
RUN cd data && python3 -m pip install .