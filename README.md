# CIFAR-10 卷积神经网络分类

基于 PyTorch 实现的轻量级 CNN，在 CIFAR-10 数据集上完成 10 类图像分类，
包含标准卷积基线、批量归一化、Dropout 正则化与数据增强，并输出测试集混淆矩阵。

一、任务简介

CIFAR-10 包含 10 类 32×32 彩色图像：飞机、轿车、鸟、猫、鹿、狗、青蛙、马、船、卡车，
训练集 50000 张、测试集 10000 张。模型目标是输入一张 32×32 彩色图像，输出其类别预测。

二、网络结构

- 特征提取：3 组卷积块（Conv3×3 + BatchNorm + ReLU + MaxPool），通道数 32 → 64 → 128
- 正则化：卷积段 Dropout（0.20 / 0.30），全连接前 Dropout 0.40
- 分类头：Flatten → Dropout → Linear(128×4×4, 10)
- 输入经随机裁剪、随机水平翻转增强，并按数据集均值/标准差归一化

三、环境依赖

| 项目 | 版本 / 说明 |
| --- | --- |
| Python | 3.10+ |
| PyTorch | 2.x（CPU / CUDA / DirectML 均可） |
| torchvision | 匹配 PyTorch 版本 |
| 依赖库 | numpy、matplotlib（可选 seaborn） |




