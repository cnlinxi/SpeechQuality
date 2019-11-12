# SpeechQuality
非侵入式（单端）客观语音评测
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