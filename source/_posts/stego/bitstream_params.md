---
title: bitstream 参数说明
date: 2024-06-15 11:48:15
tags: ffmpeg
---

|参数名称	|参数类型	|参数含义	|参数配置|
|--|--|--|--|
|i_frame_reference	|int	|B和P帧向前预测参考的帧数	| 取值范围1-16。<br/>该值不影响解码的速度，但越大解码所需的内存越大。<br/>该值影响编码速度，越大每帧编码计算的前向帧数越多，这样码率压缩效果更好。<br/>该值一般越大效果越好，但是超过6以后效果不明显。<br/>实测游戏场景下的ref分布情况，80%分布在前3帧，20分布在第10-16之间。视频会议场景建议配置在1-3之间比较合适。|
|i_dpb_size	|int|	解码缓冲区大小，最多缓冲参考帧个数个帧缓存，参数写入到sps中，指示解码段解码缓存大小。|	 |
|i_keyint_max|	int	|最大IDR帧间隔，gop_size。|	最大IDR帧间间隔，每当收到IDR帧，解码器就会清空参考队列，并且更新PPS和SPS参数。IDR帧也是一种I帧，因此，该参数如设置得比较小则更利于流畅的视频播放，但是会降低压缩效率。建议设置为帧速率的10倍。|
|i_keyint_min|	int	|最小IDR帧间隔	|该参数设置过小可能导致错误地插入IDR帧，参数限制了插入IDR帧的最小距离。建议设置等于帧速率。|
|i_scenecut_threshold|	int|	自动场景切换门限，根据其含义，表示场景变换的百分比。计算场景间的相似度，如果相似度小于该门限值则认为检测到场景切换。如果此时距离上一个IDR帧的距离小于最小IDR帧间隔，则插入一个I帧，否则插入一个IDR帧。	 |
|b_intra_refresh|	int|	是否使用周期帧内刷新替代IDR帧	 |
|i_bframe|	int	|在I帧与P帧之间可插入B帧数量（Number of B-frames）的最大值	|I帧和P帧之间的B帧数量，若设置为0则表示不使用B帧，B帧会同时参考其前面与后面的帧，因此增加B帧数量可以提高压缩比，但也因此会降低压缩的速度。|
|i_bframe_adaptive|	int|	如果为true，则自动决定什么时候需要插入B帧，最高达到设置的最大B帧数。如果设置为false,那么最大的B帧数被使用。	|B帧插入策略，该策略决定使用P帧还是B帧，<br/>0=X264_B_ADAPT_NONE（总是使用B帧），<br/> 1=X264_B_ADAPT_FAST（快速算法），<br/>2=X264_B_ADAPT_TRELLIS（最佳算法），<br/>三种算法的计算复杂度依次增加。|
|i_bframe_bias	|int|	B帧插入倾向	|影响插入B帧的倾向，越高越容易插入B帧，但是100也不能保证完全使用B帧。一般情况下不推荐修改。|
|i_bframe_pyramid	|int|	允许B帧作为参照帧。如果关闭，那么只有I帧和P帧才能作为参照帧。可以作为参照帧的B帧的量化参数会介于P帧和普通B帧之间。只在–b-frames设置大于等于2时此选项才生效。如果是在为蓝光光盘编码，请使用none或者strict。	|none:不允许B帧作为参照帧；<br/>strict:一个图像组内只允许一个B帧参照帧，这是蓝光编码强制要求的标准；<br/>normal:任意使用B帧参照帧；|
|b_open_gop|	int|	是否开启opengop功能|	1：openGop<br/>0：closeGop|
|b_bluray_compat|	int|	bluray-compat模糊兼容校准的一些参数	 |
|i_avcintra_class	|int	| 	 
|i_avcintra_flavor|	int	 |	 
|b_deblocking_filter|	int	|控制去块滤波器是否打开，推荐打开。	| 
|i_deblocking_filter_alphac0|	int|	alpha去块滤波器|	取值范围 -6 ~ 6 数字越大效果越强|
|i_deblocking_filter_beta|	int|	beta去块滤波器|	取值范围 -6 ~ 6 数字越大效果越强|
|b_cabac|	int	|使用CABAC熵编码技术,为引起轻微的编码和解码的速度损失，但是可以提高10%-15%的编码质量。|	 
|i_cabac_init_idc	|int|	给出算术编码初始化时表格的选择	 |
|b_interlaced|	int|	帧场编码	 |
|b_constrained_intra|	int|	开启SVC编码的底层要求的强制帧内预测	 |
|i_cqm_preset|	int|	自定义量化矩阵(CQM), 初始化量化模式为flat	 |
|*psz_cqm_file|	char|	读取JM格式的外部量化矩阵文件，忽略其他cqm选项|	 
|cqm_4iy[16]|	uint8_t	 	 |
|cqm_4py[16]|	uint8_t	 	 |
|cqm_4ic[16]|	uint8_t	 	 |
|cqm_4pc[16]|	uint8_t	 	 |
|cqm_8iy[64]|	uint8_t	 	 |
|cqm_8py[64]|	uint8_t	 	 |
|cqm_8ic[64]|	uint8_t	 	 |
|cqm_8pc[64]|	uint8_t	 	 |

来源

https://blog.csdn.net/CrystalShaw/article/details/90173307