---
title: H264 编码器参数设置
date: 2024-06-15 09:45:15
tags: ffmpeg
---

在 ffmpeg 源码中，libx264.c 文件中的 `X264_init` 函数中设置了编码器的参数，

```c++
// libx264.c
// X264_init 函数定义内部代码
X264Context *x4 = avctx->priv_data;
x264_param_default(&x4->params);
... // x4->params 的一些字段设置，后面初始化 SPS 和 PPS 就用到 x4->params 的值
x4->enc = x264_encoder_open(&x4->params); // 打开一个编码器，返回 x264_t 类型
```

设置 `params` 分为以下几个过程：

**# 1：默认初始化**

调用 `x264_param_default` 方法对 `params` 字段进行默认初始化，`params` 字段类型为 `x264_param_t`，定义在 x264 源码中，

```c++
// x264.h (x264库)
typedef struct x264_param_t {
    // 字段太多，这里省略。。。
}
```

`x264_param_default` 函数定义在 x264 源码中，

```c++
// base.c
void x264_param_default(x264_param_t *param) {
    memset(param, 0, sizeof(x264_param_t));
    ...
    param->i_width = 0;
    param->i_height = 0;
    param->i_fps_num = 25;
    param->i_fps_den = 1;
    param->i_level_idc = -1;
    param->i_frame_reference = 3;
    param->i_keyint_max = 250;
    param->i_keyint_min = X264_KEYINT_MIN_AUTO;//0
    param->i_bframe = 3;
    param->i_bframe_pyramid = X264_B_PYRAMID_NORMAL;//2
    // 内容太多，这里省略
}
```

**# 2: 根据 preset 和 tune 设置**

在 libx264.c 文件的 `X264_init` 函数中，默认初始化 `params` 之后，再根据 preset 和 tune 的值设置 `params`，

```c++
// libx264.c
// X264_init 函数定义内部代码

if (x4->preset || x4->tune) {
    if (x264_param_default_preset(&x4->params, x4->preset, x4->tune) < 0) {
        
    }
}
```

preset 和 tune 均为字符串，例如

```c++
av_opt_set(ctx->priv_data, "preset", "veryfast", 0);
av_opt_set(ctx->priv_data, "tune", "zerolatency", 0);
```

preset 的值可以是 ultrafast，superfast，veryfast，faster，fast，medium，slow，slower，veryslow，placebo。

tune 的值可以是 film，animation，grain，stillimage，psnr，ssim，fastdecode，zerolatency，touhou。

```c++
// x264 源码中的 base.c 文件
int x264_param_default_preset(x264_param_t *param, const char *preset, const char *tune) {
    x264_param_default(param);  // 这里再一次调用默认初始化，其实应该设置一个默认初始化标志位，避免重复调用
    if (preset && param_apply_preset(param, preset) < 0) return -1;
    if (tune && param_apply_tune(param, tune) < 0) return -1;
    return 0;
}

static int param_apply_preset(x264_param_t *param, const char* preset) {
    if (!strcasecmp(preset, "ultrafast")) {
        param->i_frame_reference = 1;
        param->b_deblocking_filter = 0;
        param->b_cabac = 0;
        param->i_bframe = 0;
        ...
    } else if (!strcasecmp(preset, "veryfast")) {
        param->i_frame_reference = 1;
        param->analyse.i_subpel_refine = 2;
        param->rc.i_lookahead = 10;
        param->analyse.b_mixed_references = 0;
        param->analyse.i_trellis = 0;
        param->analyse.i_weighted_pred = X264_WEIGHTP_SIMPLE;   // 1
    } ...
}

static int param_apply_tune(x264_param_t *param, const char* tune) {
    ...
    else if (len == 11 && !strncasecmp(tune, "zerolatency", 11)) {
        param->i_bframe = 0;
        param->rc.i_lookahead = 0;
        ...
    }
}
```

**# 3: 根据 AVCodecContext 设置**

```c++
// libx264.c
// X264_init 函数定义内部代码

if (avctx->level > 0)
    x4->params.i_level_idc = avctx->level;

x4->params.p_log_private        = avctx;
x4->params.i_log_level          = X264_LOG_DEBUG;
x4->params.i_csp                = convert_pix_fmt(avctx->pix_fmt);
x4->params.i_bitdepth           = av_pix_fmt_desc_get(avctx->pix_fmt)->comp[0].depth;

if (avctx->bit_rate) {
    x4->params.rc.i_bitrate   = avctx->bit_rate / 1000;
    x4->params.rc.i_rc_method = X264_RC_ABR; // 2
}

if (avctx->max_b_frames >= 0) {
    x4->params.i_bframe = avctx->max_b_frames;
}
```

**# 4: 根据 x4 的字段设置**

```c++
// libx264.c
// X264_init 函数定义内部代码

PARSE_X264_OPT("weightp", wpredp);
PARSE_X264_OPT("level", level);
PARSE_X264_OPT("psy-rd", psy_rd);
PARSE_X264_OPT("deblock", deblock);
PARSE_X264_OPT("partitions", partitions);
PARSE_X264_OPT("stats", stats);

if (x4->b_pyramid >= 0) // 默认值为 -1
    x4->params.i_bframe_pyramid = x4->b_pyramid;
```

这里 `PARSE_X264_OPT` 是一个宏，可以简单的理解为 

```c++
#define PARSE_X264_OPT(name, var) x264_param_parse(&x4->params, name, x4->var)
```

函数 `x264_param_parse` 根据 x4 中的字段对 params 的字段进行设置，

```c++
// x264 源码中的 base.c 文件
int x264_param_parse(x264_param_t *p, const char *name, const char *value) {
    ...
}
```

**# 5: 根据 profile 设置**

```c++
// libx264.c
// X264_init 函数定义内部代码
if (x4->profile) {
    if (x264_param_apply_profile(&x4->params, x4->profile) < 0) { }
}
```

profile 的值可以是 baseline，high，high10，high422，high444，main。设置方法为，

```c++
encoder.ctx->profile = decoder.ctx->profile;
```

这里 profile 是 int 类型，然后再改为字符串，

```c++
// libx264.c
// X264_init 函数定义内部代码
x4->profile = x4->profile_opt;
/* Allow specifying the x264 profile through AVCodecContext. */
if (!x4->profile)
    switch (avctx->profile) {
    case AV_PROFILE_H264_BASELINE:
        x4->profile = "baseline";
        break;
    case AV_PROFILE_H264_HIGH:
        x4->profile = "high";
        break;
    case AV_PROFILE_H264_HIGH_10:
        x4->profile = "high10";
        break;
    case AV_PROFILE_H264_HIGH_422:
        x4->profile = "high422";
        break;
    case AV_PROFILE_H264_HIGH_444:
        x4->profile = "high444";
        break;
    case AV_PROFILE_H264_MAIN:
        x4->profile = "main";
        break;
    default:
        break;
    }
```

`x264_param_apply_profile` 也是 x264 源码中的函数，定义位于文件 `base.c` 中。



