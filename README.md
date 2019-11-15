# SpeechQuality
无参考（单端）客观语音评测
****

双端客观语音质量评测，需要原始语音和经过损伤的语音，如PESQ[1,2], P.OLQA[3]，而单端语音质量评测只需要经损伤的语音，即可预测人类对该语音的主观评测分数。

## 目前的几个方法

- 在语音停顿点，检测该语音是否为非正常卡顿。出发点来自于，卡顿被证明是影响用户听感的最重要的因素之一[8]。保证准确率的前提下，尽可能找出可能的非正常停顿点。

- 基于规则的传统方法。如p.563[4]，ITU-T提出的用于评判VoIP语音质量的非侵入式模型。同时，还有Anique+[5].
  
- 深度学习直接预测语音的质量。如QualityNet[6], MOSNet[7]以及NISQA[8].

> [1] ITU-T Rec. P.862, “Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs,” .
> 
> [2] ITU-T Rec. P.862.2, “Wideband extension to Recom- mendation P.862 for the assessment of wideband tele- phone networks and speech codecs,” .
> 
> [3] ITU-T Rec. P863, “Perceptual objective listening qual- ity assessment,” .
>
> [4] ITU-T Rec. P.563, “Single-ended method for objective speech quality assessment in narrow-band telephony ap- plications,” .
>
> [5] D. Kim and A. Tarraf, “Anique+: A new american na- tional standard for non-intrusive estimation of narrow- band speech quality,” Bell Labs Technical Journal, vol. 12, no. 1, pp. 221–236, Spring 2007.
>
> [6] Szu wei Fu, Yu Tsao, Hsin-Te Hwang, and Hsin- Min Wang, “Quality-net: An end-to-end non-intrusive speech quality assessment model based on BLSTM,” in Proc. Interspeech 2018, 2018, pp. 1873–1877.
>
> [7] Lo C C, Fu S W, Huang W C, et al. MOSNet: Deep Learning based Objective Assessment for Voice Conversion[J]. arXiv preprint arXiv:1904.08352, 2019.
>
> [8] Mittag G, Möller S. Non-intrusive Speech Quality Assessment for Super-wideband Speech Communication Networks[C]//ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019: 7125-7129.

## 目前进展

- 语音停顿点+规则判断非正常停顿(QualityBasedVad)。目前在第一批的PLC数据集上可以做到准确率100%，查全率50%，但是似乎遇到瓶颈，进一步优化有一定难度。
  目前计划进一步在更大的测试集上测试，以确保该方法的可行性。

- p.563(p563NewProject)。在ITU-T p.563的基础上，结合目前给到的测试集，进一步提升该算法的性能。

- MOSNet(MOSNet)，QualityNet(QualityNet)的模型正在构造模型过程中，尚没有出来具体的性能指标。计划先用公开的数据集进行训练，用自己实际的数据集微调，尽可能提升该方法的性能，以用于备选。

## 迭代

目前在进行的工作：  

1. 根据MOSNet论文中报道的内容，利用公开数据集，尝试复现该方法。如果成功，接下来用自己的数据集微调，以期获得更好的性能；
2. 调优p.563，以期在给到的数据集上，差的音频和好的音频，MOS分能够拉开差距，更好的适用实验室给到的数据集。
3. 在新数据集上，测试上周的算法的各项指标。

### 2019年11月13日
昨日工作：
完成了MOSNet模型的初步测试，使用自己标注的数据集进行了简单的训练，今天计划搜集数据集训练一份baseline模型。

今日计划：

1. 搜集一份较大的数据集。无论是传统方法还是深度学习，都需要一份比较大的数据集，目前主要有两条路，一条自己精细标注，另外一条看看网络上有没有现成的数据集可供下载。
2. 今天计划继续标注新来的音频数据，并且给出上周给出方案的评测数据。关于实时性，流式检测，目前只能想到是每隔固定时间，如5s启动一次检测。
3. 利用搜集的数据集，训练深度学习的模型。

### 2019年11月14日
利用公开的语音转换的数据集进行了初步的训练。整个模型如下：

```

Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         [(None, None, 257)]       0         
_________________________________________________________________
reshape_4 (Reshape)          (None, None, 257, 1)      0         
_________________________________________________________________
conv2d_48 (Conv2D)           (None, None, 257, 16)     160       
_________________________________________________________________
conv2d_49 (Conv2D)           (None, None, 257, 16)     2320      
_________________________________________________________________
conv2d_50 (Conv2D)           (None, None, 86, 16)      2320      
_________________________________________________________________
conv2d_51 (Conv2D)           (None, None, 86, 32)      4640      
_________________________________________________________________
conv2d_52 (Conv2D)           (None, None, 86, 32)      9248      
_________________________________________________________________
conv2d_53 (Conv2D)           (None, None, 29, 32)      9248      
_________________________________________________________________
conv2d_54 (Conv2D)           (None, None, 29, 64)      18496     
_________________________________________________________________
conv2d_55 (Conv2D)           (None, None, 29, 64)      36928     
_________________________________________________________________
conv2d_56 (Conv2D)           (None, None, 10, 64)      36928     
_________________________________________________________________
conv2d_57 (Conv2D)           (None, None, 10, 128)     73856     
_________________________________________________________________
conv2d_58 (Conv2D)           (None, None, 10, 128)     147584    
_________________________________________________________________
conv2d_59 (Conv2D)           (None, None, 4, 128)      147584    
_________________________________________________________________
time_distributed_2 (TimeDist (None, None, 512)         0         
_________________________________________________________________
time_distributed_3 (TimeDist (None, None, 64)          32832     
_________________________________________________________________
dropout_1 (Dropout)          (None, None, 64)          0         
_________________________________________________________________
frame (TimeDistributed)      (None, None, 1)           65        
_________________________________________________________________
avg (GlobalAveragePooling1D) (None, 1)                 0         
=================================================================
Total params: 522,209
Trainable params: 522,209
Non-trainable params: 0
```
在上述数据集上可以获得的性能指标(使用第一批给到的音频数据，手工切分并标注MOS，利用其中的13段音频进行测试)：

```
[UTTERANCE] Test error(MSE)= 1.394297
[UTTERANCE] Linear correlation coefficient= 0.383357
[UTTERANCE] Spearman rank correlation coefficient= 0.469559
```

![](https://github.com/cnlinxi/SpeechQuality/blob/master/figures/MOSNet_distribution.png)
![](https://github.com/cnlinxi/SpeechQuality/blob/master/figures/MOSNet_scatter_plot.png)

**目前遇到的最大困难在于数据集，太少了**
1. 昨天整理了上次说到的ITU-T p_sup数据集，这份数据集的缺点在于，好的音频和差的音频差别不大，MOS=1的音频和MOS=5的音频是有差别，但是我听起来，感觉就是音量不同，有点杂音，变化不大。昨天整理出来，也拿这份数据集训练了，性能很不好。
2. 我把到手的自己的数据集都拿过来，今天是否要整理自己的数据集？这很耗时，因为要长音频切分，要一个个标注MOS。现在初步切分了一下，有1000多条音频。

### 2019年11月15日
早会记录：

1. 动手标注具体的音频数据。首先使用VAD切分，然后对每一段音频数据整体给定一个主观评分。
2. 上报数据采用一种直方图的形式，先VAD切分音频数据，然后类似于类似于NetEQ中的IAT制作成直方图，上报。
3. 请教别人问题要明确，描述要清楚。给出问题的上下文，描述出这个问题是什么。