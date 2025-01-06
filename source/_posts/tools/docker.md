---
title: 容器使用示例
date: 2021-11-26 16:23:48
tags: docker
p: tools/
---
介绍简单的容器使用示例。
<!--more-->
# All in Docker
To check all docker images, you can type
```
sudo docker images
```
The output of this command may be
```
REPOSITORY                                                            TAG               IMAGE ID       CREATED        SIZE
tensorflow/tensorflow                                                 latest-gpu        edb49f6a133b   2 days ago     5.53GB
...
```
> tensorflow/tensorflow is an image provided by `tensorflow` official organization. It's is actually a linux OS and the tensorflow library is installed systematicly.

Take the first image as an example. This image is used to provide an environment for deep learning. Now we show how to clone this image to make a highly customized and suitable for ourselves deep-learning enrivonment, and we can keep modifing this new cloned image without influncing the original image. Input the following command,
```
sudo docker tag tensorflow/tensorflow:latest-gpu tensorflow/tensorflow:public
```
By using another tag name, we create a new image. Please type a command to see what has happened,
```
sudo docker images
```
and the output will be
```
REPOSITORY                                                            TAG               IMAGE ID       CREATED        SIZE
tensorflow/tensorflow                                                 latest-gpu        edb49f6a133b   2 days ago     5.53GB
tensorflow/tensorflow                                                 public            edb49f6a133b   18 hours ago   5.53GB
```
Whenever want to do deep learning in future, we should first run this new image, and then train or evaluate data in our own enviromment. So, how to run an image?
```
sudo nvidia-docker run -p 6010:6006 -p 9527:22 -v /home/shajianjian/work:/workspace -itd --name tf-public  tensorflow/tensorflow:public
```
Be aware that we use `nvidia-docker` instead of `docker` because `nvidia-docker` can help us to load `cuda` related components automatically when lauching this deep learning environment, otherwise, we cannot use `GPU` accelarating learning.

We may have a great experience with our deep learning environment for a couple of days, but afterwards, we find that some matured deep learning projects are implemented with PyTorch, which is not existed in our environment, how to do?

Don't worry, be happy!

Login our environment and change current directory to

```
ssh domain-user@192.168.5.116

cd /workspace
```

Install miniconda, and then install pytorch by conda(details are ignored because we don't focus on it in this artical). They say that all modications in `/workspace` are saved and even when reboot our deep learning environment(re-run our image). Suppose we install pytorch systematicly, like tensorflow, then we should save it as a new image(marked it as B, and our first cloned image is marked as A), or else the pytorch library will be disgarded when stop image `A`. Let's see how to save A as B after changing A systematicly,

```
sudo docker commit tf-public localharbor.xxxx.com/tf/public:v1
```

where `tf-public` is the name of container when we run A, and A has a tag of `public`. Use the above command, we commit our changes and re-save it as the new image B with a tag of `public-v1`. Notice that iamge `A` is not changed, our changes maked to `A` are saved in `B`, i.e. `B`=`A`+`changes`.

`localharbor.xxxx.com/tf/public:v1` can be substituted by any other value, but we use it here because we will push the image `B` to localharbor which is an online image repository, and anyone else can pull it(image `B`) to use directly without install tensorflow and pytorch again. This process saves his/her time.

Login harbor and then pull this image,
```
sudo docker login localharbor.xxxx.com
sudo docker push localharbor.xxxx.com/tf/public:v1
```

Take my favorite DL framework pytorch as an example, the whole steps are:

1.
print all running containers
```
sudo docker ps
```
the printing msg is:
```
CONTAINER ID   IMAGE                                                                  COMMAND                  CREATED         STATUS       PORTS                                          NAMES
0f3dc93c8ebc   localharbor.xxxx.com/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel   "/bin/bash"              3 weeks ago     Up 3 weeks   0.0.0.0:9528->9528/tcp, 0.0.0.0:9527->22/tcp   pytorch
209def4f9cd0   scrapinghub/splash                                                     "python3 /app/bin/sp…"   5 weeks ago     Up 5 weeks   0.0.0.0:8050->8050/tcp                         thirsty_wu
5917c29d675e   localharbor.xxxx.com/tf/pml:tf-remote-latest                       "/bin/bash"              10 months ago   Up 5 weeks   0.0.0.0:23->22/tcp, 0.0.0.0:6009->6006/tcp     tf-remote
```

2.
print all images
```
sudo docker images
the printing msg is

localharbor.xxxx.com/pytorch/pytorch                              1.7.1-cuda11.0-cudnn8-devel      d0d89d27be2a   10 months ago   13.2GB
...(omitted)
```
3.
Exec the following commands in sequence order
```
sudo docker commit pytorch localharbor.xxxx.com/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel_latest
sudo docker login localharbor.xxxx.com
sudo docker push localharbor.xxxx.com/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel_latest
```

# PyTorch
Specially, please execute the following command to launch a `PyTorch` environment,
```
sudo docker run -itd --gpus all --runtime=nvidia -p 9527:22 -p 9528:9528 --shm-size 8G -v /home/shajianjian/pytorch:/workspace --name pytorch --hostname pytorch localharbor.xxxx.com/pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
```


where the path and port mappings should be replaced with your own values.

Maybe you should first start the `ssh` service,
```
# if not install ssh service, please run this command first
# apt install openssh-server

service ssh start
```
Check the status of `ssh`,
```
service ssh status
```

reset the password for the user `root` on the docker os:
```
passwd
# operate according to hints
```

If cannot connect to os by `ssh root@192.158.5.116 -p 9527` and then input the correct password, please modify the `/etc/ssh/sshd_config` by adding the following line:
```
PermissionLoginRoot yes
```

This virtual system has a default `root` user whose password is `pytorch`, so let's login this virtual system by
```
> ssh root@192.168.5.116 -p 9527
input password: pytorch
(base) root@pytorch:/workspace#
```

进入容器

```sh
docker exec -it [容器id或容器名称] /bin/bash
```

Enjoy it~