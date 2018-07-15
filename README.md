# Qbot_project
CNN for classification and quantification of water content in different substances. 
All hyperspectral inages and training data excluded from this repository due to large files sizes. 
Download them from OneDrive if don't have them.

# Installation
Login to GCS instance (put your project-id)

```gcloud compute --project "<project-id>" ssh --zone "us-west1-b" "qbot"```

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

Run automatic setup script (Nvidia cuda driver, Docker CE, nvidia-docker2)

`curl https://raw.githubusercontent.com/alekseykonoiko/Qbot_project/master/GCS/install-gpu.sh | bash`

Run this command to check if nvidia-docker2 is running

`sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`

Add persistent disk with `gcloud` tool

`gcloud compute disks create [DISK_NAME] --size 20GB --type pd-ssd` 

Attach created persistance disk to running/stopped instance

`gcloud compute instances attach-disk [INSTANCE_NAME] --disk [DISK_NAME]`

df -h
sudo lsblk

Upload file to GCS bucket storage

`gsutil cp Desktop/kitten.png gs://my-awesome-bucket`

Download an object from your bucket

`gsutil cp gs://my-awesome-bucket/kitten.png Desktop/kitten2.png`

Copy an object to a folder in the bucket

`gsutil cp gs://my-awesome-bucket/kitten.png gs://my-awesome-bucket/just-a-folder/kitten3.png`

List the contents at the top level of your bucket:

`gsutil ls gs://my-awesome-bucket`



