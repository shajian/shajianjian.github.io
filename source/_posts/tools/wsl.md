---
title: wsl 子系统
date: 2024-05-22 10:35:48
tags: wsl
p: tools/
---


```sh
# 查看可以在线安装的 wsl 系统版本
wsl --list --online
# 安装一个 wsl 发行版
wsl --install Ubuntu-20.04
# 登录子系统
wsl -d Ubuntu-20.04

# 登录默认子系统，例如 Ubuntu
wsl
```

**# 查看当前机器上已经安装的 wsl 系统**

```sh
wsl -l -v
# 或者使用下面的，都一样
wsl --list --verbose
```

**# 默认系统**

```sh
# 查看默认子系统
wslconfig /list
# 修改默认子系统
wslconfig /setdefault Ubuntu-20.04
```

# conda 错误总结

```
Error while loading conda entry point: conda-libmamba-solver (libarchive.so.19: cannot open shared object file: No such file or directory)
```

原因是这些包必须来自同一通道。
解决方案：
```
conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
```


# pytorch 多机训练

有两台 windows 系统，均启动了 wsl。如果有 linux 系统，那就省事多了，但是在 wsl 内执行 pytorch 多机训练，会出现一些问题。

首先 wsl 系统内 ip 与 windows 宿主并不相同，这使得即使 windows 机器在同一局域网内，两个 wsl 也无法相互通信。

## 设置 wsl 与 windows 使用同一 ip

在用户目录下创建 `.wslconfig` 文件，并在其中输入内容，以启动网络镜像模式，

```sh
[experimental]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

然后关闭 wsl

```sh
wsl --shutdown
```

然后开启 wsl 应用配置

```sh
wsl
```

然后运行以下命令查看 ip

```sh
ip address  # wsl
ipconfig    # windows
```

## 防火墙设置

打开防火墙设置，将两台机器上的公共网络的防火墙关闭。

## 查看 GPU 通信方式

```sh
nvidia-smi topo -m

nvidia-smi nvlink --status
```


# WSL 端口转发

```sh
netsh interface portproxy add v4tov4 listenport=<yourPortToForward> listenaddress=0.0.0.0 connectport=<yourPortToConnectToInWSL> connectaddress=(wsl hostname -I)
```

在此示例中，需要更新 <yourPortToForward> 到端口号，例如 listenport=4000。 
listenaddress=0.0.0.0 表示将接受来自任何 IP 地址的传入请求。 
侦听地址指定要侦听的 IPv4 地址，可以更改为以下值：IP 地址、计算机 NetBIOS 名称或计算机 DNS 名称。 
如果未指定地址，则默认值为本地计算机。 需要将 <yourPortToConnectToInWSL> 值更新为希望 WSL 连接的端口号，例如 connectport=4000。 
最后，connectaddress 值必须是通过 WSL 2 安装的 Linux 分发版的 IP 地址（WSL 2 VM 地址），可通过输入命令：wsl.exe hostname -I 找到。

# 防火墙

启用和禁用Windows防火墙

```sh
Set-NetFirewallProfile -All -Enabled false
Set-NetFirewallProfile -All -Enabled true
```

若要使用 cmd 命令

```sh
netsh advfirewall set allprofiles state off

# 查看监听状态
netsh interface portproxy show all
```

创建新的防火墙规则

```sh
# powershell
New-NetFirewallRule -DisplayName "Allow WSL2 Port 46096" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 46096
```

# 释放磁盘空间

wsl2 中删除文件后，windows 磁盘空间并未释放，要释放 windows 磁盘空间，需要

1. 找到如下文件

```sh
C:\Users\admin\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx
# C:\Users\admin\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx
```

2. 打开 powershell，执行

```sh
wsl --shutdown
# wsl --shutdown Ubuntu-20.04
diskpart    # 打开管理计算机的驱动器 DiskPart 命令
select vdisk file="C:\Users\admin\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
compact vdisk
detach vdisk
```
