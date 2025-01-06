---
title: glibc 版本
date: 2024-07-27 11:06:44
tags: c++
p: cpp/glibc_version
---

# 1. 查看 glibc

```sh
ldd --version
# 或者
strings /lib/libc.so.6 | grep GLIBC
```

查看 glibc++

```sh
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

# 2. 升级 glibc

centos7 默认的 glibc 版本太低，需要升级。

## 2.1 升级 make

当前 make 版本是 3.82

```sh
make -v
```

需要升级到 4.x

```sh

wget https://mirrors.aliyun.com/gnu/make/make-4.3.tar.gz
 
# 4.解压压缩包并建立构建目录
tar -xf make-4.3.tar.gz
cd make-4.3
mkdir build
cd build
 
# 5.指定安装到具体的目录下，此示例表示将make安装到/opt下
../configure --prefix=/opt/make
 
# 6.编译安装
make && make install

mv /usr/bin/make /usr/bin/make.bak
# 7.建立软连接
ln -s /opt/make/bin/make /usr/bin/make
```

## 2.2 升级 gcc

```sh
# 1.安装升级依赖
yum install -y gcc-c++ glibc-devel mpfr-devel libmpc-devel gmp-devel glibc-devel.i686
​
# 2.下载gcc9.3.1安装包
cd /backup
wget https://ftp.gnu.org/gnu/gcc/gcc-9.3.0/gcc-9.3.0.tar.gz --no-check-certificate
​
# 3.解包并执行编译前的准备
tar -xf gcc-9.3.0.tar.gz
cd gcc-9.3.0
# 下载依赖包
./contrib/download_prerequisites
# 建立构建目录
mkdir build
# 进入构建目录
cd build
​
# 4.指定安装到具体的目录下，此示例表示将make安装到/usr下(说明：若安装到非/usr目录,如安装到/opt/gcc，则在编译完成后需要配置环境变量、建立软连接。)
../configure --enable-checking=release --enable-language=c,c++ --disable-multilib --prefix=/usr
​
# 5.编译安装
make -j4 # -j代表编译时的任务数，一般有几个cpu核心就写几，构建速度会更快一些。该步骤执行时间很长
make install
​
# 6.上一步若安装目录不是/usr，则需要在编译完成后配置环境变量、建立软连接，若为/usr目录则跳过此步骤
# 假设将gcc编译安装到了/opt/gcc目录
# 6.1配置环境变量
vi /etc/profile.d/gcc.sh
# gcc环境配置
export PATH=/opt/gcc/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc/lib
# 编辑完成后:wq保存并退出
# 6.2重载环境变量
source /etc/profile
# 6.3重新生成新的链接
# 取消原始链接
unlink /usr/bin/cc
# 建立新链接
ln -sf /opt/gcc/bin/gcc /usr/bin/cc
ln -sf /opt/gcc/lib/gcc/x86_64-pc-linux-gnu/9.3.0/include /usr/include/gcc
# 设置库文件
echo "/opt/gcc/lib64" >> /etc/ld.so.conf.d/gcc.conf
# 加载动态连接库
ldconfig -v
# 查看加载结果
ldconfig -p | grep gcc
​
# 7.安装完成后检查gcc版本，若gcc升级失败则需查找失败原因并重新进行升级操作
gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/libexec/gcc/x86_64-pc-linux-gnu/9.3.0/lto-wrapper
Target: x86_64-pc-linux-gnu
Configured with: ../configure --enable-checking=release --enable-language=c,c++ --disable-multilib --prefix=/usr
Thread model: posix
gcc version 9.3.0 (GCC)
```

## 2.3 升级 glibc

```sh
# 1.查看glibc函数库版本
strings /lib64/libc.so.6 | grep -E "^GLIBC" | sort -V -r | uniq
# 输出
GLIBC_2.2.5
GLIBC_2.2.6
GLIBC_2.3
GLIBC_2.3.2
GLIBC_2.3.3
GLIBC_2.3.4
GLIBC_2.4
GLIBC_2.5
GLIBC_2.6
GLIBC_2.7
GLIBC_2.8
GLIBC_2.9
GLIBC_2.10
GLIBC_2.11
GLIBC_2.12
GLIBC_2.13
GLIBC_2.14
GLIBC_2.15
GLIBC_2.16
GLIBC_2.17  # 当前最高版本
GLIBC_PRIVATE
 
# 2.下载glibc-2.31安装包
cd /backup
wget https://mirrors.aliyun.com/gnu/glibc/glibc-2.31.tar.gz
 
# 3.进入到解压目录
tar -xf glibc-2.31.tar.gz
cd glibc-2.31
 
# 4.查看安装glibc的前提依赖，对于不满足的依赖需要进行升级，使用yum -y install xxx 升级或安装即可
cat INSTALL | grep -E "newer|later" | grep "*"
# 输出
* GNU 'make' 4.0 or newer
* GCC 6.2 or newer
* GNU 'binutils' 2.25 or later
* GNU 'texinfo' 4.7 or later
* GNU 'bison' 2.7 or later
* GNU 'sed' 3.02 or newer
* Python 3.4 or later
* GDB 7.8 or later with support for Python 2.7/3.4 or later
* GNU 'gettext' 0.10.36 or later
# 假设上述依赖条件已全部满足
 
# 5.建立构建目录，执行编译安装
mkdir build
 
# 6.指定安装到具体的目录下，此示例表示将make安装到/opt下
cd build/
../configure  --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin --disable-sanity-checks --disable-werror
 
# 7.编译安装
make -j4  # 此处时间较长
make install
# 解决新启动远程终端时报一个WARNING
make localedata/install-locales
 
# install结束会出现一个错误，此错误可忽略
# 错误输出
Execution of gcc -B/usr/bin/ failed!
The script has found some problems with your installation!
Please read the FAQ and the README file and check the following:
- Did you change the gcc specs file (necessary after upgrading from
  Linux libc5)?
- Are there any symbolic links of the form libXXX.so to old libraries?
  Links like libm.so -> libm.so.5 (where libm.so.5 is an old library) are wrong,
  libm.so should point to the newly installed glibc file - and there should be
  only one such link (check e.g. /lib and /usr/lib)
You should restart this script from your build directory after you've
fixed all problems!
Btw. the script doesn't work if you're installing GNU libc not as your
primary library!
make[1]: *** [Makefile:120: install] Error 1
make[1]: Leaving directory '/backup/glibc-2.31'
make: *** [Makefile:12: install] Error 2'
 
# 8.安装完成后检查glibc版本
strings /lib64/libc.so.6 | grep -E "^GLIBC" | sort -V -r | uniq
# 输出
GLIBC_PRIVATE
GLIBC_2.30
GLIBC_2.29
GLIBC_2.28
GLIBC_2.27
GLIBC_2.26
GLIBC_2.25
GLIBC_2.24
GLIBC_2.23
GLIBC_2.22
GLIBC_2.18
GLIBC_2.17
GLIBC_2.16
GLIBC_2.15
GLIBC_2.14
GLIBC_2.13
GLIBC_2.12
GLIBC_2.11
GLIBC_2.10
GLIBC_2.9
GLIBC_2.8
GLIBC_2.7
GLIBC_2.6
GLIBC_2.5
GLIBC_2.4
GLIBC_2.3.4
GLIBC_2.3.3
GLIBC_2.3.2
GLIBC_2.3
GLIBC_2.2.6
GLIBC_2.2.5
```