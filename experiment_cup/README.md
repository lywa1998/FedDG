# 实验
本目录主要进行了眼底图像视盘和视杯的实验。因为本人是第一次接触眼底图像这类图像分割数据集，
因此本目录主要是理解这些概念和结果可视化的代码。
process.py 是查看眼底图像视盘和视杯是如何进行分割得到其轮廓。
visual.py 是利用 process 中学到的分割轮廓来生成 FedDG 预测的结果和真值的对比，如![](./test-0.png)。