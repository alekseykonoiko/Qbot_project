# Qbot_project
CNN for classification and quantification of water content in different substances. 
All hyperspectral inages and training data excluded from this repository due to large files sizes. 
Download them from OneDrive if don't have them.

# Installation

Install essential Ubuntu packages
```
sudo apt-get update

sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common
```
```
curl https://raw.githubusercontent.com/alekseykonoiko/Qbot_project/master/GCS/install-gpu.sh | bash
```
Run this command to check if nvidia-docker2 is running

```
sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```