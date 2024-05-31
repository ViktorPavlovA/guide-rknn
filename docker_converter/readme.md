# How to used

## Step 1

`cd ~/ && mkdir docker_folder`

`cp <your_yolov8.pt> ~/docker_folder/`

`docker  build . -t builder`

**if error with pull add this:**

`nano /etc/docker/daemon.json`

```
{
    "log-opts": {
        "max-size": "10m"
    },
    "registry-mirrors": ["https://mirror.gcr.io", "https://daocloud.io", "https://c.163.com/", "https://registry.docker-cn.com"]
}  

```

`sudo systemctl restart docker.service`

## Step 2

`docker run -it --name builder -v  ~/docker_folder:/pt builder`

`sh convert.sh`

`exit`

## Step 3

**Check ~/docker_folder/**

To used again do this command before start: `docker rm builder`

