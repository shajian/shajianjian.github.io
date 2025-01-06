---
title: rknn 数据准备说明
date: 2024-09-04 09:12:14
tags: rknn
summary: rknn 如何准备输入数据？
---

本文以 3588 为例说明。

# 1. pass_through=1

rknn 数据直通即设置 pass_through=1，可以避免推理库内部对数据做各种预处理，简单来说，就是 `rknn_inputs_set` 执行速度变快。

以一个实例进行说明。

pytorch 模型输入 shape 为 `(n, c, h, w)`，需要注意 `w` 需要 16 字节对齐，即 w 是 16 的倍数。转为 rknn 模型后，输入 shape 为 `(n, h, w, c)`，这里我感觉 rknn 内部还是对输入数据进行了 layout 转换：nhwc->nchw，虽然 rk 开发文档中有这么一句话：

> pass_through 如果设置为 1，则输入的数据直接输入到模型内进行运算，不进行如何转换;

（笔者注：上一句话中“如何”应该改为“任何”）

我使用了一个超简单模型，

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x)
        return x

    def save2onnx(self, path):
        self.eval()
        dummy_input = torch.randn(1, 3, 2, 16)  # (n, c, h, w)
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(self, dummy_input, path, verbose=True, 
                          input_names=input_names,
                          output_names=output_names, opset_version=11)
```

在转为 rknn 模型后，我使用输入数据 `[1,2, ..., 96]`，但是输出数据变成 `[1, 4, 7, ..., 94, 2, 5, 8, ..., 95, 3, 6, 9, ..., 96]`，显然，这是做了 nhwc -> nchw 的 layout 转换。由于 pass_through=1，故这里我的输入实际上已经转为 float16 类型。注意，实验均使用 C++ 完成。

我又搞了一个实验，将模型改为

```python
def forward(self, x):
    return torch.mean(x, [2, 3], keepdim=True)
```

rknn 将输入进行 layout 转换，然后计算平均，结果为 `[[[[47.5]], [[48.5]], [[49.5]]]]`，shape 为 `(1, 3, 1, 1)`。

rknn 模型的输入是 nhwc，输出是 nchw。


# 2. zero copy

零拷贝输入输出 api，需要的模型输入宽度 w 也需要是 16 的整数。