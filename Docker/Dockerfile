
ARG cuda_version=9.0
ARG cudnn_version=7
ARG ubuntu_version=16.04
ARG TENSORFLOW_VERSION=0.12.1
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.2.0

FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel-ubuntu${ubuntu_version}

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system packages

RUN apt-get update && apt-get -y install \
      	bc \
		build-essential \
		cmake \
		curl \
		g++ \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		pkg-config \
		python3-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		python3-dev \
		python3-tk \
		python3-numpy \
		&& \
      apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* 

# Install pip
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
# 	python get-pip.py && \
# 	rm get-pip.py

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py
# Add SNI support to Python
# RUN pip3 --no-cache-dir install \
# 		pyopenssl \
# 		ndg-httpsclient \
# 		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-nose \
		python3-h5py \
		python3-matplotlib \
		python3-sklearn \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install \
		Cython \
		path.py \
		Pillow \
		wheel

# Install TensorFlow
RUN pip3 --no-cache-dir install --upgrade tensorflow-gpu

# Install Keras
RUN pip3 --no-cache-dir install keras

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006

WORKDIR "/root"
CMD ["/bin/bash"]