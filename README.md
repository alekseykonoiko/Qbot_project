# Qbot_project
CNN for classification and quantification of water content in different substances. 
All hyperspectral inages and training data excluded from this repository due to large files sizes. 
Download them from OneDrive if don't have them.

# Virtual Machine (VM) Instance Creation Setup
Create an instance in Compute Engine with the following settings:
Region: us-west1 (or any other region with GPU based servers
Machine Type: 4 vCPU
Number of GPU: 1 (or any available)
GPU Type: Nvidia Tesla K80 (or better)
Boot disk: Ubutu 16.04 LTS
Boot disk type: SSD
Size: 20 GB (or more)
Access scopes: Allow full access to all Cloud APIs
Firewall: Allow HTTP traffic, Allow HTTPS

# Installation
Throughout installation process replace all words wrapped by <....> with your own setting. Also try to follow my exact files and naming structure to avoid errors when executing commands.
## Ubuntu system setup
Login into Google Cloud Services (GCS) instance SSH terminal

`gcloud compute --project "<project-id>" ssh --zone "us-west1-b" "qbot"`

In case of login error, click on the arrow next to SSH and View gcloud command 

Run to install essential Ubuntu packages
```
`sudo apt-get update
 sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common`
```

Run automatic setup script (instals: Nvidia cuda driver, Docker CE, nvidia-docker2)

`curl https://raw.githubusercontent.com/alekseykonoiko/Qbot_project/master/GCS/install-gpu.sh | bash`

Run this command to check if nvidia-docker2 is running

`sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`

Configure the Google Cloud firewall to allow your local client to connect to port 7007 on your VM.

`gcloud compute firewall-rules create tensorboard --allow tcp:7007`
## Storage and Docker setup procedures

### Shared storage section

Create bucket storage in your GCS account, follow this tutorial

`https://cloud.google.com/storage/docs/creating-buckets`

Using GUI create folder `qbot` in bucket storage and upload this files to the folder:
-   CNN_server.py
-   Dockerfile
-   training_data.npz

Also create folder `logs` inside `qbot` 

From SHH terminal create directory for `gcsfuse` mount

`mkdir "$(pwd)"/shared`

Run this in SSH to mount bucket storage to bind mount "shared"

`gcsfuse <bucket_name> "$(pwd)"/shared`

Resove docker permissions

`sudo usermod -a -G docker $USER`

Exit SSH (type `exit`) and log in again

`gcloud compute --project "<project-id>" ssh --zone "us-west1-b" "qbot"`

Current directory to qbot folder

`cd shared/qbot`

If permission problem encountered, switch to root bash mode

`sudo -i`

Now `cd` again

`cd "$(pwd)"/shared/qbot`

Run `ls` to verify that all the files including `Dockerfile` are in the folder

### Docker section

Build docker image

 `docker build -t qbot_docker .`

After unmount and remove shared volume
```
cd
sudo umount shared/
sudo rm -r shared
```

Run docker container with bind mount "shared"

`docker run --runtime=nvidia --rm -it --name qbot_container -p 7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Mount storage bucket again in second SSH terminal

`sudo gcsfuse <bucket_name> "$(pwd)"/shared`

Exit docker (type `exit`) container and run again

`docker run --runtime=nvidia --rm -it --name qbot_container -p 7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Now `cd` to shared directory in docker container

`cd shared/qbot`

Run python script

`python3 <script-name>.py`

From now you can use Google Cloud Storage GUI to upload new scripts for training and execute them with `python3` as described above

### Cycle of stopping instance and restarting when not used

Open two SSH terminals and run docker container in the first ternimal

`docker run --runtime=nvidia --rm -it --name qbot_container -p 7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

On second terminal mount bucket storage again (it unmounts when VM instance stops)

`sudo gcsfuse <bucket_name> "$(pwd)"/shared`

Now exit docker (type `exit`) container and run again

`docker run --runtime=nvidia --rm -it --name qbot_container -p 7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Now file system ready for work, `cd` to qbot folder as before and proceed with training

## Useful commands and tools

### VIM editor

To open file in editor type

`vim <filename>`

In editor there are two modes `INSERT` mode and default mode (vim loades in default mode). `INSERT` mode allows you to edit file and default mode allows to enter commands like save or quite.

To enter `INSERT` mode you should press `Shift + I` to exit press `Esc`
 
To save file and exit vim editor, in default mode press `Shift + :` and type `x` and press `Enter`

### Bucket storage

Upload file to GCS bucket storage

`gsutil cp Desktop/image.png gs://bucket-name`

Download an object from your bucket

`gsutil cp gs://my-bucket/image.png Desktop/image.png`

Copy an object to a folder in the bucket

`gsutil cp gs://my-bucket/image.png gs://my-bucket/just-a-folder/image.png`

List the contents at the top level of your bucket:

`gsutil ls gs://my-bucket`

### Docker
List all active containers (including stopped)

`docker ps -a`

Start stopped container

`docker exec -it qbot_container bash`

Print info about container

`docker inspect qbot_container`

Remove stopped docker container

`docker rm -f qbot_container`

### Bash in SSH
Remove directory in bash run

`rm -r <dir-name>`

Login to root bash terminal

`sudo -i`

Path to current directory

`pwd`

List current directory contents

`ls`


