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
sudo apt-get update
sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common
```

Run automatic setup script (instals: Nvidia cuda driver, Docker CE, nvidia-docker2)

`curl https://raw.githubusercontent.com/alekseykonoiko/Qbot_project/master/GCS/install-gpu.sh | bash`

Run this command to check if nvidia-docker2 is running

`sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`

Configure the Google Cloud firewall to allow your local client to connect to port 7007 on your VM (don't run this if it was used before).

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

Exit (type `exit`) your SSH and login again

`gcloud compute --project "<project-id>" ssh --zone "us-west1-b" "qbot"`

`cd` to qbot `dir`

`cd "$(pwd)"/shared/qbot`

Run `ls` to verify that all the files including `Dockerfile` are in the folder

### Docker section

Build docker image

 `docker build -t qbot_docker .`
 
 Unmount bucket storage
 
 ```
 cd
 umount shared/
 rm -r shared/
 ```

Run docker container with bind mount "shared"

`docker run --runtime=nvidia --rm -it --name qbot_container -p 0.0.0.0:7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Exit docker container (type `exit`) and mount bucket storage again

`sudo gcsfuse <bucket_name> "$(pwd)"/shared`

Run docker container again

`docker run --runtime=nvidia --rm -it --name qbot_container -p 0.0.0.0:7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Now `cd` to shared directory in docker container

`cd shared/qbot`

Run python script

`python3 <script-name>.py`

From now you can use Google Cloud Storage GUI to upload new scripts for training and execute them with `python3` as described above

### Cycle of stopping instance and restarting when not used

In SSH terminal mount bucket storage again (it unmounts when VM instance stops)

`sudo gcsfuse <bucket_name> "$(pwd)"/shared`

Then tun docker container again

`docker run --runtime=nvidia --rm -it --name qbot_container -p 0.0.0.0:7007:6006 -v "$(pwd)"/shared:/root/shared qbot_docker bash`

Now file system ready for work, `cd` to qbot folder as before and proceed with training

### Connect to tensorboard
#### Real time tensorboard

Run second SSH terminal and login to running docker container 

`docker exec -it qbot_container bash`

In docker container `cd` to `shared/qbot/logs/` folder and run

`tensorboard --logdir ./ --host 0.0.0.0 --port 6006`

Now note your VM instance exernal ip (you can find it in details about qbot instance in GCS GUI). Then enter following address in browser

http://<external_ip>:7007

## After training tensorboard 
`cd` to `shared/qbot/logs/

`tensorboard --logdir ./ --host 0.0.0.0 --port 6006`

Open in browser

http://<external_ip>:7007

## Useful commands and tools

### VIM editor

To open file in editor enter

`vim <filename>`

There are two modes in vim editor: `INSERT` mode and default mode (vim loades in default mode). `INSERT` mode allows you to edit the file and default mode allows to enter commands like save or quite.

To enter `INSERT` mode you should press `Shift + I` to exit press `Esc`
 
To save file and exit vim editor, in default mode press `Shift + :` then enter `x` and press `Enter`

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

Path to current directory

`pwd`

List current directory contents

`ls`


