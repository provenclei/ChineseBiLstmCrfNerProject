# Bi-LSTM NER PROJECT

> Chinese NER Based Bi-LSTM, TensorFlow Implementations.
>
> 基于Bi-Lstm的命名实体识别，使用TensorFlow实现。

## Requirement

* tensorflow = 1.13.1
* numpy = 1.15.0
* python = 3.6 
* jieba == 0.42.1

## Result

Average Precision:  90.56%

Average Recall:  90.33%

Average F1:  90.44%

Loss:  0.099449

tag\eval | precision | recall | FB1 |
:-: | :-: | :-: | :-: | :-:
LOC | 91.93% | 92.45% | 92.19% |
ORG | 85.38%| 84.99% | 85.18% |
PER | 93.95%| 92.44% | 93.19% |
