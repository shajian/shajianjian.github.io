---
title: FFmpeg 源码分析
date: 2023-12-18 13:48:15
tags: ffmpeg
---

**# 1 创建编码器**

```c++
const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
```

此时 `libx264.c` 文件中的变量

```c++
FFCodec ff_libx264_encoder = { 
    .p.name = "libx264",
    CODEC_LONG_NAME("libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 30"),
    .p.type = AVMEDIA_TYPE_VIDEO,
    .p.id = AV_CODEC_ID_H264,
    .p.capabilities = ...,
    .p.priv_class = &x264_class,
    .p.wrapper_name = "libx264",
    .priv_data_size = sizeof(X264Context),
    .init = X264_init,
    ...
};
```

实际是返回的就是 `ff_libx264_encoder` 对象的第一个字段 `p` 的地址（实际上也是 `ff_libx264_enoder` 的地址）

这是因为 `FFCodec` 结构体中第一个字段类型就是 `AVCodec`，而方法

```c++
const AVCodec *av_codec_iterate(void **opaque) {
    uintptr_t i = (uintptr_t)*opaque;   // i 初始值为 0
    const FFCodec *c = codec_list[i];   // 遍历 codec_list 中的对象
    if (c) {
        *opaque = (void*)(i+1);         // codec_list 的 index 前进一位
        return &c->p;                   // 返回 AVCodec 类型的地址
    }
}
```

**# 2 创建编码器上下文对象**


```c++
AVCodecContext* ctx = avcodec_alloc_context3(codec);
```

源码是

```c++
AVCodecContext *avcodec_alloc_context3(const AVCodec *codec) {
    AVCodecContext *avctx = av_malloc(sizeof(AVCodecContext));  // 分配对象内存
    if (!avctx) { return NULL; }
    if (init_context_defaults(avctx, codec) < 0) {
        av_free(avctx);
        return NULL;
    }
    return avctx;
}
```

这里最重要的就是初始化函数，初始化 AVCodecContext 到底设置了什么，

```c++
// options.c 
static int init_context_defaults(AVCodecContext *s, const AVCodec *codec) {
    // 直接类型强转即可，因为指向同一个地址
    const FFCodec *const codec2 = ffcodec(codec);
    int flags = 0;
    memset(s, 0, sizeof(AVCodecContext));

    s->av_class = &av_codec_context_class;  // 见下方代码

    // AVMEDIA_TYPE_VIDEO
    s->codec_type = codec ? codec->type : AVMEDIA_TYPE_UNKNOWN;
    if (codec) {
        s->codec = codec;
        s->codec_id = codec->id;
    }

    ...
    // flags = AV_OPT_FLAGS_VIDEO_PARAM;
    av_opt_set_defaults2(s, flags, flags);
    ... // 后面在继续分析，这里先暂停，跳转去分析 av_opt_set_defaults2 这个函数
}

static const AVClass av_codec_context_class = {
    .class_name = "AVCodecContext",
    .item_name = context_to_name,
    .option = avcodec_options,
    ...
}
```

`av_opt_set_defaults2` 顾名思义就是设置一些默认值，我们看下设置了什么，

```c++
const AVOption *av_opt_next(const void *obj, const AVOption *last) {
    const AVClass *class;
    if (!obj) return { NULL; }
    // 这里 obj 就是 AVCodecContext 指针

    // 类型强转，那肯定是两个指针指向同一个地址，见下方 AVCodecContext 结构体定义
    class = *(const AVClass**)obj;
    if (!last && class && class->option && class->option[0].name)
        return class->option;
    if (last && last[1].name)
        return ++last;
    return NULL;
}
```

上面代码中 `class->option` 位于 `options_table.h` 中，包含了所有的选项设置，这是一个数组，
所以上面代码中后续返回的是这个数组中下一个选项，即这是一个遍历数组中选项的函数。

```c++
typedef struct AVCodecContext {
    const AVClass *av_class;
    ...
}
```

```c++
void av_opt_set_defaults2(void *s, int mask, int flags) {
    const AVOption *opt = NULL;
    // 传入参数：AVCodecContext 指针，上一个 AVOption 指针（初始时上一个为 NULL）
    while ((opt = av_opt_next(s, opt))) { // 遍历所有选项
        
    }
}
```

所有选项见下方定义，创建一个编码器时，这些选项的值决定了编码过程，所以弄清楚这些编码选项非常重要。

```c++
// options_table.h
static const AVOption avcodec_options[] = {
{"b", "set bitrate (in bits/s)", OFFSET(bit_rate), AV_OPT_TYPE_INT64, {.i64 = AV_CODEC_DEFAULT_BITRATE }, 0, INT64_MAX, A|V|E},
...
}
```

关于选项的说明可以参见[官方文档](https://ffmpeg.org/ffmpeg-codecs.html#Codec-Options) ，不完全一样，因为官方文档给的是 `ffmpeg` 这个命令的选项，但是可以参考一下。

例如，选项 `b` 表示设置比特率，

```sh
ffmpeg -b:v 1000k   # 设置视频流比特率
ffmpeg -b:a 128k    # 设置音频流比特率
```

编码相关参数设置，参考 [Encode/H.264](https://trac.ffmpeg.org/wiki/Encode/H.264)

回到 avcodec_options 中的选项定义中，第一个选项设置比特率，其中

```c++
OFFSET(bit_rate) -> offsetof(AVCodecContext, bit_rate)
```
计算的是 `bit_rate` 这个成员变量在 `AVCodecContext` 中的偏移量，结果是 `56`，即偏移了 56 个 byte。

至此，我们已经知道 `av_opt_set_defaults2` 就是将选项的默认值设置到 `AVCodecContext` 对应的各个变量上，接着我们再看 `init_context_defaults` 这个方法还做了哪些事情，

```c++
// options.c 
static int init_context_defaults(AVCodecContext *s, const AVCodec *codec) {
    const FFCodec *const codec2 = ffcodec(codec);   // 类型强转，因为指向同一个地址
    ...

    s->time_base            = (AVRational){0, 1};   // 时间基，默认值为 0
    s->framerate            = (AVRational){0, 1};   // 帧率，默认值为 0
    ...
    s->pix_fmt              = AV_PIX_FMT_NONE;
    ...

    if (codec && codec2->priv_data_size) {
        // 分配内存空间并全 0 初始化这个内存空间，以 x264 为例，这个空间对应的是 X264Context
        // 见 ff_libx264_encoder 定义
        s->priv_data = av_mallocz(codec2->priv_data_size);
        if (!s->priv_data) return AVERROR(ENOMEM);
        if (codec->priv_class) {
            // 根据优先级，先指针成员选择，然后类型强转，所以是将 AVCodecContext 的 priv_data
            // 从 void* （实际上是 X264Context*）转为 AVClass*，而 AVClass 是 X264Context
            // 第一个成员变量的类型，所以指向同一个地址，可以强转，故其实是设置 X264Context
            // 的第一个成员变量 class 为 x264_class
            *(const AVClass**)s->priv_data = codec->priv_class;
            // 设置下方的 `options` 选项（即 preset tune profile 等）的默认值到 X264Context 的变量上
            av_opt_set_defaults(s->priv_data);
        }
    }
    if (codec && codec2->defaults) {
        int ret;
        const FFCodecDefault *d = codec2->defaults; // 设置各个编码器自带的一些默认值
                                                    // 见下方的 `x264_defaults` 变量定义
        while (d->key) {
            ret = av_opt_set(s, d->key, d->value, 0);// 设置到 AVCodecContext 中
            av_assert0(ret >= 0);
            d++;
        }
    }
    return 0;
}
```

以 libx264 为例，上述代码中 `codec->priv_class` 就是 `ff_libx264_encoder` 中的 `&x264_class`，此变量定义如下，

```c++
static const AVClass x264_class = {
    .class_name = "libx264",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};
```

其中 `options` 这个变量定义在 `libx264.c` 文件中，此变量是一个数组，包含了一些 x264 自有的一些选项设置，与上面 `avcodec_options` 的格式类似，

```c++
static const AVOption options[] = {
    { "preset", "Set the encoding preset", OFFSET(preset), AV_OPT_TYPE_STRING, { .str = "medium" }, 0, 0, VE},
    { "tune", "Tune the encoding params", OFFSET(tune), AV_OPT_TYPE_STRING, {0}, 0, 0, VE},
    { "profile", "Set profile restrictions", OFFSET(profile_opt), AV_OPT_TYPE_STRING, {0}, 0, 0, VE},
    ...
};
```

但是这里的选项不是设置到 `AVCodecContext` 中的变量，而是设置到 `X264Context` 的变量上，所以，`X264Context` 的 `preset` 变量默认值为 `"medium"`，而 `tune` 和 `profile` 两个变量的默认值为 `""`。

```c++
// libx264.c

// 这些选项默认值将设置到 AVCodecContext 中的相应变量上去。
static const FFCodecDefault x264_defaults[] = {
    { "b",          "0" },  // "b" 是比特率，见 avcodec_options 的中的选项定义
    { "bf",         "-1" },
    { "flags2",     "0" },
    ...
    { "qmin",       "-1" },
    { "qmax",       "-1" },
    { "qblur",      "-1" },
    { "qcomp",      "-1" },
    ...
}
```
所以这里将编码器的比特率又重新设置为 `0` 。


至此，完成了 `AVCodecContext` 的对象创建以及成员变量的默认赋值。

## 2. 关联上下文

```c++
// avcodec.c
int avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options) {
    AVCodecInternal *avci;
    const FFCodec *codec2;

    codec2 = ffcodec(codec);
    avctx->codec_type = codec->type;
    avctx->codec_id = codec->id;
    avctx->codec = codec;

    avci = av_codec_is_decoder(codec) ?
        ff_decode_internal_alloc() :
        ff_encode_internal_alloc();

    avctx->internal = avci;
    avci->buffer_frame = av_frame_alloc();      // 分配内部 frame 缓存
    avci->buffer_pkt = av_packet_alloc();       // 分配内部 packet 缓存

    // 将 width 和 height 分别设置到 coded_width 和 coded_height
    ret = ff_set_dimensions(avctx, avctx->width, avctx->height);

    if (av_codec_is_encoder(avctx->codec))
        ret = ff_encode_preinit(avctx); // 这里面又做了很多 validating 和初始化工作
    else
        ret = ff_decode_preinit(avctx);

    ...

    if (!(avctx->active_thread_type & FF_THREAD_FRAME) ||
        avci->frame_thread_encoder) {// frame_thread_encoder 有效，见 ff_frame_thread_encoder_init 函数
        if (codec2->init) {
            lock_avcodec(codec2);
            ret = codec2->init(avctx);  // init 方法为 X264_init，参见 libx264.c 文件
            unlock_avcodec(codec2);
        }
    }
    ...
}
```

上述代码设置了编码器的一些参数，包括分配内部的 frame 和 packet 缓存空间，设置帧的宽高，以及多线程编码上下文对象等。与 x264 相关的初始化工作则在 `X264_init` 中完成，

```c++
static int X264_init(AVCodecContext *avctx) {
    X264Context *x4 = avctx->priv_data;
    x264_param_default(&x264->params);

    x4->params.b_deblocking_filter = avctx->flags & AV_CODEC_FLAG_LOOP_FILTER;

    if (x4->preset || x4->tune) {
        if (x264_param_default_preset(&x4->params, x4->preset, x4->tune) < 0) {
            ...
        }
    }
}
```

以上代码中，`x264_param_default` 用于设置一些默认参数值，具体可参见 x264 源码中的 `base.c` 文件。[x264 git 仓库地址](https://code.videolan.org/videolan/x264.git)

`x264_param_default_preset` 则根据 `preset` 和 `tune` 选项值来设置参数。

**# 根据 preset 设置参数**

代码位于 x264 源码中的 `base.c` 文件的 `param_apply_preset` 函数。

**# 根据 tune 设置参数**

代码位于 x264 源码中的 `base.c` 文件的 `param_apply_tune` 函数。

**# 根据 profile 设置参数**

```c++
// libx264.c  X264_init 函数
x4->profile = x4->profile_opt;  // 这里 profile_opt 默认值为 ""，见本文件 options 变量
if (!x4->profile) {
    switch (avctx->profile) {   // 见 options_table.h 中的 avcodec_options 变量
        ...
    }
}

if (x4->profile) {
    // 此函数位于 x264 源码的 base.c 文件
    if (x264_param_apply_profile(&x4->params, x4->profile) < 0) { ... }
}
```


## 3. 编码

```c++
// encode.c

int avcodec_send_frame(AVCodecContext* avctx, const AVFrame* frame) {
    AVCodecInternal *avci = avctx->internal;
    ...
    
    if (avci->buffer_frame->buf[0]) // 输入在当前状态不被接受，try again
        return AVERROR(EAGAIN);
    if (!frame) {
        avci->draining = 1;     // 对应最后的 null frame 冲刷编码器
    } else {
        ret = encode_send_frame_internal(avctx, frame);
        if (ret < 0) return ret;
    }
    
    if (!avci->buffer_pkt->data && !avci->buffer_pkt->side_data) {  // 内部缓存的编码后包数据为 NULL，则调用下方函数进行编码
        ret = encode_receive_packet_internal(avctx, avci->buffer_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
            return ret;     // 编码出错
    }
    ...
}
```

此方法对编码器上下文进行了判断，例如是否成功 open，是否是编码器，是否进入 draining 模式等。对于一个进来的正常的帧，然后内部调用 `encode_send_frame_internal` 方法，但是这个方法除了将待编码的帧数据拷贝到内部缓存中，其他也没做什么事情。

```c++
static int encode_send_frame_internal(AVCodecContext *avctx, const AVFrame* src) {
    AVCodecInternal *avci = avctx->internal;
    AVFrame *dst = avci->buffer_frame;
    int ret;
    ret = av_frame_ref(dst, src);   // 将 src 指向的数据拷贝到 dst 指向的对象
    if (ret < 0) return ret;

    if (avctx->codec->type == AVMEDIA_TYPE_VIDEO) {
        ret = encode_generate_icc_profile(avctx, dst);
        if (ret < 0) return ret;
    }
    if (!(avctx->flags & AV_CODEC_FLAG_FRAME_DURATION)) // 由于flags默认值为0（参见默认选择设置 avcodec_options）
        dst->duration = 0;
}
```

记住拷贝的帧数据存储在 `avci->buffer_frame` 中。然后就是对拷贝到缓存的帧数据进行编码，方法定义如下，

```c++
// encode.c
// 编码，编码后数据存储到 AVPacket 中
static int encode_receive_packet_internal(AVCodecContext* avctx, AVPacket* avpkt) {
    ...
    int ret;
    if (ffcodec(avctx->codec)->cb_type == FF_CODEC_CB_TYPE_RECEIVE_PACKET) { // false
        ret = ffcodec(avctx->codec)->cb.receive_packet(avctx, avpkt);
        ...
    } else {
        // 执行这个分支
        ret = encode_simple_receive_packet(avctx, avpkt);
    }
    int ret;
    if (ffcodec(avctx->codec)->cb_type == FF_CODEC_CB_TYPE_RECEIVE_PACKET) { // false
        ret = ffcodec(avctx->codec)->cb.receive_packet(avctx, avpkt);
        ...
    } else {
        // 执行这个分支
        ret = encode_simple_receive_packet(avctx, avpkt);
    }
}

static int encode_simple_receive_packet(AVCodecContext *avctx, AVPacket *avpkt) {
    int ret;

    while (!avpkt->data && !avpkt->side_data) { // data 和 side_data 均为 NULL，则继续循环
        ret = encode_simple_internal(avctx, avpkt);
        if (ret < 0) return ret;
    }
    return 0;
}

static int encode_simple_internal(AVCodecContext *avctx, AVPacket *avpkt) {
    AVCodecInternal *avci = avctx->internal;
    AVFrame *frame = avci->in_frame;
    const FFCodec *const codec = ffcodec(avctx->codec);
    int got_packet;
    int ret;
    ...
    if (!frame->buf[0] && !avci->draining) {    // frame 没有数据，并且没有进入 draining 状态
                                                // 记住送入的帧数据拷贝到 avci->buffer_frame
        av_frame_unref(frame);  // 对 avci->in_frame 解旧引用，准备引用新的帧数据
        ret = ff_encode_get_frame(avctx, frame);// 将 avci->buffer_frame 关联到 avci->in_frame 上
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }
    ...
    got_packet = 0;
    if (CONFIG_FRAME_THREAD_ENCODER && avci->frame_thread_encoder)
        ret = ff_thread_video_encoder_frame(avctx, avpkt, frame, &got_packet);//线程编码实现
    else
        ret = ff_encode_encode_cb(avctx, avpkt, frame, &got_packet);
}
```

我们先说明上述代码 `encode_receive_packet_internal` 中执行哪个分支：

在 libx264.c 这个文件中，`ff_libx264_encoder` 的定义中有

```c++
// libx264.c
FFCodec ff_libx264_encoder = {
    ...
    FF_CODEC_ENCODE_CB(X264_frame),
};

// codec_internal.h
#define FF_CODEC_ENCODE_CB(func)                          \
    .cb_type           = FF_CODEC_CB_TYPE_ENCODE,         \
    .cb.encode         = (func)
```

故 `cb_type` 不等于 `FF_CODEC_CB_TYPE_RECEIVE_PACKET` 。

然后 `encode_simple_internal` 这个函数中，编码实现有两个分支，其中第一个分支判断 `CONFIG_FRAME_THREAD_ENCODER` 这个宏定义为 1，`avci->frame_thread_encoder` 是否定义呢？

我们先看 `avci` 这个字段，它在 avcodec.c 文件中的 `avcodec_open2` 函数中初始化，相关代码片段如下，

```c++
// 我们这里先讨论编码器
// ff_encode_internal_alloc 函数就是分配 EncodeContext 这个结构体大小的内存
avci = av_codec_is_decoder(codec) ? ff_decode_internal_alloc() : ff_encode_internal_alloc();
avctx->internal = avci;
...
if (av_codec_is_encoder(avctx->codec))
    ret = ff_encode_preinit(avctx);     // 这个函数内部调用 ff_frame_thread_encoder_init 函数
...
```

简单跳转一下就发现，

```c++
av_cold int ff_frame_thread_encoder_init(AVCodecContext *avctx) {
ThreadContext *c;
...
if(avctx->thread_count <= 1)
    return 0;
c = avctx->internal->frame_thread_encoder = av_mallocz(sizeof(ThreadContext));
}
```

简单明了，如果编码器的 thread_count 设置为 1，那么不进行多线程编码，否则进行多线程编码，此时 `frame_thread_encoder` 有定义。

现在让我们回到前面 `encode_simple_internal` 这个函数中的分支判断，如果编码器 thread_count 为 1，执行 if 分支，否则执行 else 分支。这里我们假设 thread_count 为 1，即，执行 `ff_encode_encode_cb` 函数进行帧编码。

```c++
// encode.c
int ff_encode_encode_cb(AVCodecContext *avctx, AVPacket *avpkt,
                        AVFrame *frame, int *got_packet) 
{
    const FFCodec *const codec = ffcodec(avctx->codec);
    int ret;

    ret = codec->cb.encode(avctx, avpkt, frame, got_packet);
    ...
}
```

上面的 `codec` 前面讲过，就是 `libx264.c` 文件中的 `ff_libx264_encoder` 变量，其 `.cb.encode` 由宏调用 `FF_CODEC_ENCODE_CB(X264_frame)` 给出，即 `X264_frame` 方法。

```c++
// libx264.c
static int X264_frame(AVCodecContext *ctx, AVPacket *pkt, const AVFrame *frame,
                      int *got_packet)
{
    X264Context *x4 = ctx->priv_data;
    x264_nal_t *nal;
    int nnal, ret;
    x264_picture_t pic_out = {0}, *pic_in;
    int pict_type;
    int64_t wallclock = 0;
    X264Opaque *out_opaque;

    ret = setup_frame(ctx, frame, &pic_in); // 将 frame 的数据设置到 pic_in 中
    do {
        if (x264_encoder_encode(x4->enc, &nal, &nnal, pic_in, &pic_out) < 0)
            return AVERROR_EXTERNAL;
        ...
    } while (!ret && !frame && x264_encoder_delayed_frames(x4->enc));
    // while 判断：如果执行 OK，ret=0，并且 frame=NULL（最后冲刷的数据），并且编码器内部还有缓存的帧
    //  那么继续进行帧编码，直到  执行出错 ret!=0，或者编码器内部没有缓存帧了，就退出 while 循环
}
```

上述代码中，真正执行帧编码的是 `x264_encoder_encode` 函数，这个函数的定义位于 `x264` 项目[源码](https://code.videolan.org/videolan/x264.git) 的 `encoder.c` 文件中，这个函数定义有点庞大，可以慢慢研究。

这里，我对 SPS 和 PPS 感兴趣，所幸在 `x264_encoder_encode` 中发现了如下代码，

```c++
// (x264) encoder.c
int x264_encoder_encode(x264_t* h, x264_nal_t **pp_nal, int *pi_nal,
    x264_picture_t *pic_in, x264_picture_t* pic_out) {
    ...
    if (h->fenc->b_keyframe) {  // 对于关键帧，需要加上 SPS PPS
        nal_start(h, NAL_SPS, NAL_PRIORITY_HIGHEST);
        x264_sps_write(&h->out.bs, h->sps);
        ...
        nal_start(h, NAL_PPS, NAL_PRIORITY_HIGHEST);
        x264_pps_write(&h->out.bs, h->sps, h->pps);
    }
}
```

`x264_t` 的定义位于 `common.h` 文件（依然是 x264 项目源码，以下除非特别说明，否则均是指 x264 的项目源码）中，

```c++
struct x264_t {
    ...
    x264_sps_t sps[1];
    x264_pps_t pps[1];
};
```

先以 SPS 为例。这里 `x264_sps_write` 的函数定义位于 `set.c` 文件中，易知这是将 `h->sps` 数据写入 `h->out.bs` 中，那么 `h->sps` 数据怎么来的呢？也就是说，编码器是如何生成 SPS 的？

让我暂时先回到 ffmpeg 源码中，在前面第 2 节 “关联上下文” 中，函数 `avcodec_open2` 打开编码器，这个函数内部调用了 `codec2->init(avctx);` 语句，这里 `init` 字段表示一个函数，字段设置位于 `libx264.c` 文件中的 `ff_libx264_encoder` 中，即 `init` 的值为 `X264_init`，所以起始是调用 `X264_init` 进行一些初始化工作，这个函数前面也讲过它做的一些工作，但是这里我们再看它做的一个初始化工作：

```c++
// libx264.c
// X264_init 函数定义内部代码
X264Context *x4 = avctx->priv_data;
x264_param_default(&x4->params);
... // x4->params 的一些字段设置，后面初始化 SPS 和 PPS 就用到 x4->params 的值
x4->enc = x264_encoder_open(&x4->params); // 返回 x264_t 类型
```

上面代码中，`priv_data` 在调用 `avcodec_alloc_context3` 获取编码器 AVCodecContext 时，其内部调用 `init_context_defaults`， 见上文第 1 节内容，这个函数内部对 `priv_data` 进行了赋值，并设置了一些默认参数。

这里我们先着重看如何初始化参数 `params` 的，这很重要，影响了后面编码过程，`params` 字段类型为 `x264_param_t`，定义在 x264 源码中，

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

在 `x264_init` 函数中，调用 `x264_param_default` 进行默认初始化 `params` 之后，还设置了 `params` 的其他字段，这些其实都是使用 `AVCodecContext` 的一些字段修改 `params` 中相应的字段。

关于 `params` 更多的设置，可以参考 [H264 编码器参数设置](2024/06/15/stego/ffmpeg_x4_params)。


设置了 `x4->params` 之后，便调用 `x264_encoder_open` （这是 x264 里面的函数）打开一个编码器，这个编码器是 x264 库中实际进行编码的一个工具实例，由这个函数打开 x264 的编码器，因为 ffmpeg 实际上就是使用 x264 的编码器，所以我们再次回到 x264 的项目源码。

```c++
// encoder.c
x264_t* x264_encoder_open(x264_param_t* param, void* api) {
    x264_t *h;
    ...
    memcpy(&h->param, param, sizeof(x264_param_t));
    ...
    x264_sps_init(h->sps, h->param.i_sps_id, &h->param);
    x264_sps_init_scaling_list(h->sps, &h->param);
    x264_pps_init(h->pps, h->param.i_sps_id, &h->param, h->sps);
    ...
}
```

以上几个函数的定义为，

```c++
// encoder/set.c
void x264_sps_init(x264_sps_t *sps, int i_id, x264_param_t *param) {
    ...
}

void x264_pps_init(x264_pps_t *pps, int i_id, x264_param_t *param, x264_sps_t *sps) {
    ...
}

void x264_sps_init_scaling_list(x264_sps_t *sps, x264_param_t *param) {
    ...
}
```

我们看下 sps 的初始化函数 `x264_sps_init` 的定义，其中部分字段的设置为，

**i_num_ref_frames**

num_ref_frames 规定了可能在视频序列中任何图像帧间预测的解码过程中用到的短期参考帧和长期参考帧、互补参考场对以及不成对的参考场的最大数量。num_ref_frames 字段也决定了 8.2.5.3 节规定的滑动窗口操作的大小。num_ref_frames 的取值范围应该在 0 到 MaxDpbSize （参见 A.3.1 或 A.3.2 节的定义）范围内，包括 0 和MaxDpbSize。


```c++
// x264_sps_init 函数定义内部

sps->vui.i_num_reorder_frames = param->i_bframe_pyramid ? 2 : param->i_bframe ? 1 : 0;
sps->vui.i_max_dec_frame_buffering = 
sps->i_num_ref_frames = X264_MIN(X264_REF_MAX, X264_MAX4(param->i_frame_reference, 1 + sps->vui.i_num_reorder_frames, param->i_bframe_pyramid ? 4 : 1, params->i_dpb_size));
sps->i_num_ref_frames -= param->i_bframe_pyramid == X264_B_PYRAMID_STRICT/*1*/;
if (param->i_keyint_max == 1) { // i_keyint_max：最大IDR帧间隔，gop_size。
    sps->i_num_ref_frames = 0;
    sps->vui.i_max_dec_frame_buffering = 0;
}
```

我们在代码中打印出计算 `i_num_ref_frames` 所用到的几个字段的值，

```c++
// 打印 encoder.ctx 的一些参数进行分析
X264Context* x4 = (X264Context*)encoder.ctx->priv_data;
x264_param_t& p = x4->params;
printf("i_bframe_pyramid: %d, i_bframe: %d, i_frame_reference: %d, i_dpb_size: %d\n", p.i_bframe_pyramid, p.i_bframe, p.i_frame_reference, p.i_dpb_size);
```

结果为，

```sh
i_bframe_pyramid: 2, i_bframe: 3, i_frame_reference: 3, i_dpb_size: 0, max_b_frame: 0
```

在前面 `x264_param_default` 函数定义中，我们知道 `i_bframe_pyramid` 和 `i_bframe` 为默认值 `2` 和 `3`，`i_frame_reference` 默认值为 `3`，如果设置了 preset 为 veryfast，那么 `i_frame_reference` 值改为 `1`，参见 [H264 编码器参数设置](2024/06/15/stego/ffmpeg_x4_params)。


字段含义：参考文章 [bitstream 参数说明](/2024/06/15/stego/bitstream_params)。

如果想要将 `i_bframe, i_bframe_pyramid` 均设置为 0，



## preset 几个值的性能比较

参见 [文章](https://write.corbpie.com/ffmpeg-preset-comparison-x264-2019-encode-speed-and-file-size/)

## ffmpeg 源码安装

https://blog.csdn.net/wushakun/article/details/136352544

## x264 源码安装

https://blog.csdn.net/m0_52146047/article/details/109710349

## SPS 参数含义

https://blog.csdn.net/CrystalShaw/article/details/129375250