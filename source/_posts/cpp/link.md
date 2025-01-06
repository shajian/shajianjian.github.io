---
title: 链接
date: 2021-06-08 11:06:44
tags: c++
p: cpp/link
---
## 动态库目录
1. 加入 `/lib`, `/usr/lib` 等默认搜索路径
2. 设置 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR LIB PATH>`
3. 修改配置文件 `/etc/ld.so.conf`，然后执行 `ldconfig`
4. `gcc` 添加选项 `-Wl,-rpath=<YOUR LIB PATH>`。这个路径会保存到程序中。
以上是动态（运行时）链接器（也可称为加载器）寻找动态库的搜索目录。


### -Wl
这个参数表示后面的参数传递给链接器 `ld`

### -rpath
添加一个目录到运行库搜索路径，可以使用 `$ORIGIN`，它表示执行文件所在的目录，注意在 Makefile 中需要写为 `$$ORIGIN`。

### --as-needed
链接器参数，表示仅链接其 symbol 在 binary 中用到的库。默认是开启这个 flag，若要关闭，使用 `--no-as-needed`，例如

```c++
// main.c
void foo();

int main(void) {
    foo();
    return 0;
}
```

```c++
// lib/foo.c
#include <stdio.h>

void foo() { printf("foo\n"); }
```

```c++
// lib/bar.c
#include <stdio.h>

void bar() { printf("bar\n"); }
```

```makefile
# Makefile
main: main.c lib/libfoo.so lib/libbar.so
    gcc -o main main.c -Wl,--no-as-needed,-rpath='$$ORIGIN/lib' -L. -lfoo -lbar

lib/libfoo.so: lib/foo.c
    gcc -fPIC -shared -o lib/libfoo.so lib/foo.c

lib/libbar.so: lib/bar.c
    gcc -fPIC -shared -o lib/libbar.so lib/bar.c
```

查看 `main` 链接的库，
```shell
$ ldd main
```

## 动态库优先级

|优先级（1 最高）|路径|
|--|--|
|1|编译时指定链接的动态库路径 RPATH|
|2|环境变量 LD_LIBRARY_PATH 指定的路径|
|3| RUNPATH|
|4|/etc/ld.so.conf 指定的路径|
|5| /lib|
|6|/usr/lib|

## 查看 -Wl, rpath 设置的库路径

```sh
readelf -d your_library.so | grep PATH
```

简单来说，RPATH就是在LD_LIBRARY_PATH之前，会优先让执行档去寻找相应的动态库，当然了有的操作系统支持RUNPATH的话，会在RUNPATH设置情况下自动忽略RPATH,而会先去寻找LD_LIBRARY_PATH之后再去着RUNPATH,(注意这里面的顺序关系，RUNPATH在LD_LIBRARY_PATH之后,而其会让RPATH忽略，但RPATH却在LD_LIBRARY_PATH之前)，相关顺序这里简单交代下： RPATH   --   LD_LIBRARY_PATH -- RUNPATH(出现会屏蔽RPATH) -- /etc/ld.so.conf -- builtin dircetories(/lib, /usr/lib)。

CMake在默认情况下是会给你的 exe 加入相关 RPATH 的。CMake里面维护了3个比较重要的RPATH变量，即CMAKE_SKIP_RPATH,CMAKE_SKIP_BUILD_RPATH,CMKAE_INSTALL_RPATH.

对于第一个变量CMAKE_SKIP_RPATH，强制CMake不在构建期间和安装install期间给你加上它所认为的RPATH.

即
```sh
cmake .. -DCMAKE_SKIP_RPATH=TRUE
```

第二个和第三个变量也比较简单，就是分别在构建期间和安装期间不允许CMake给你加入相关RPATH

```
cmake .. -DCMAKE_SKIP_BUILD_RPATH=TRUE
cmake .. -DCMAKE_SKIP_INSTALL_RPATH=TRUE
```

当然了，如果你之后想要追加RPATH,只需要对这三个变量设置成FALSE就可以了。

## LIBRARY_PATH 与 LD_LIBRARY_PATH 区别

LIBRARY_PATH:

- LIBRARY_PATH环境变量用于指定编译器在链接阶段搜索库文件的路径。
- 当编译器在链接时需要查找共享库时，首先会在LIBRARY_PATH中指定的路径中搜索库文件，然后再在系统默认的搜索路径中查找。
- 通过设置LIBRARY_PATH，可以覆盖系统默认的搜索路径，优先使用指定路径中的库文件。

LD_LIBRARY_PATH:

- LD_LIBRARY_PATH环境变量用于指定运行时可执行文件在加载共享库时搜索库文件的路径。
- 当可执行文件在运行时需要加载共享库时，会根据LD_LIBRARY_PATH中指定的路径搜索库文件。
- 通过设置LD_LIBRARY_PATH，可以指定程序运行时搜索共享库的路径，并优先使用指定路径中的库文件。

## SONAME

共享库的文件名，以 `math` 库为例，如下

```sh
lib + math + .so (+ major version number)
```

（以下这段内容来自文章 [Linux C/C++动态链接库如何版本管理？](https://zhuanlan.zhihu.com/p/554118491)）

Linux通过版本号来管理动态库的版本，版本号最多有3级，其格式为libname.so.x.y.z

- x: major release，非兼容修改，可能对接口做了大改动，比如重命名、增加或减少参数等。
- y: minor release，不改变兼容性，但是增加了新接口
- z: patch release，不改变兼容性，仅仅是修复bug、或者优化代码实现、优化性能等。

思考一个问题，程序编译时所依赖的某个库的版本是如何决定的？我们知道编译是可通过-lhello选项指定要依赖hello这个动态库，此时其实链接器ld只会傻乎乎在在搜索目录下查找是否存在libhello.so这个文件，而不管他的版本是多少。

当libhello.so找到之后ld就可以进行链接过程了，此刻ld会查看libhello.so内部的SONAME字段，比如是libhello.so.1，那么ld之后的可执行程序在运行时就会依赖libhello.so.1，这个程序在运行时就需要去搜索目录下查找是否存在libhello.so.1这个文件。

如果动态库没有SONAME字段，则程序依赖的是动态库原始的名字，既形如libhello.so

> SONAME是编译生成动态链接库时通过soname指定的一个字段，会被写入so的文件当中，用于管理版本，通过形如gcc -Wl,-soname,libhello.so.1 的方式指定。

所以上述hello库完整的编译命令为：gcc -fPIC -g -Wl,-soname,libhello.so.1 -shared -o libhello.so.1.0.2 hello.c
可通过 objdump -p libhello.so.1.0.2 | grep SONAME 查看库中标识的版本号。