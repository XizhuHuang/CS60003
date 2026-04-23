# CS60003
深度学习与空间智能课程PJ

# EuroSAT_MLP 🌍

基于 **三层全连接神经网络（MLP）** 的遥感图像分类项目，在 **EuroSAT 数据集** 上进行训练与评估，并结合可视化与误差分析对模型行为进行深入分析。

## 📌 项目简介

本项目从零实现了一个多层感知机（MLP），使用 **NumPy** 完成前向传播、反向传播及参数更新，并应用于遥感图像分类任务。

主要内容包括：

- 三层全连接神经网络实现
- 手写反向传播与优化器（SGD）
- EuroSAT 数据集分类（10类）
- 混淆矩阵分析
- 权重可视化
- 类别模板（Class Template）可视化
- 错例分析（Error Analysis）


## 📁 项目结构

```
EUROSAT_MLP/
│
├── data/                  # 数据加载（不包含原始数据）
│   └── eurosat.py
│
├── models/                # 模型结构
│   ├── mlp.py
│   ├── layers.py
│   ├── activations.py
│   └── loss.py
│
├── optim/                 # 优化器
│   └── sgd.py
│
├── train/                 # 训练相关
│   ├── trainer.py
│   └── metrics.py
│
├── search/                # 超参数搜索
│   └── grid_search.py
│
├── visualization/         # 可视化代码
│
├── outputs/               # 模型输出（权重等）
│
├── test.py                # 测试 & 错误分析
├── main.py                # 训练入口
├── main.ipynb             # 实验Notebook
│
├── batchsize.py           # batch size实验
├── LRdecay.py             # 学习率衰减实验
├── regVIS.py              # 正则化可视化
│
└── README.md

```


## 📊 数据集说明

本项目使用 **EuroSAT_RGB** 数据集：

- 共 **27,000 张遥感图像**
- 每张图像尺寸：`64 × 64 × 3`
- 共 **10 个类别**：

```

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

```

⚠️ **注意：仓库中未包含数据集**

## 📥 数据集下载

请自行下载 EuroSAT 数据集，并按如下结构放置：

```

data/EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── ...

```

推荐下载地址：

- 官方：https://github.com/phelber/eurosat


## 🧠 模型结构

三层全连接网络：

```

Input (12288)
↓
FC (1024) + ReLU
↓
FC (512) + ReLU
↓
FC (10)
↓
Softmax

````


## 🏋️‍♂️ 训练方式

运行：

```bash
python main.py
````

或使用 Notebook：

```bash
main.ipynb
```


## 📦 预训练模型

已训练好的最佳模型权重（推荐直接使用）：

🔗 **百度网盘下载：**

> 链接: [https://pan.baidu.com/s/1Efg9Fwo3rYhkc9thmAi4cA](https://pan.baidu.com/s/1Efg9Fwo3rYhkc9thmAi4cA)
> 提取码: `wb53`

下载后放入：

```
outputs/best_model_2.npz
```


## 🧪 测试与评估

运行：

```bash
python test.py
```

输出内容：

* Test Accuracy
* Confusion Matrix
* 高置信错误样本
* 典型混淆样本可视化

## 📈 实验结果

* Test Accuracy ≈ **68.2%**

### 主要发现：

* 森林（Forest）、工业区（Industrial）识别较好
* 易混淆类别：

  * Residential ↔ Industrial
  * Highway ↔ River
  * HerbaceousVegetation ↔ PermanentCrop


## 🔍 可视化分析

### 1️⃣ Hidden Layer Weights

* 第一层权重可视化结果呈现 **噪声状分布**
* 说明 MLP 未学习到局部空间结构

### 2️⃣ Class Template

通过优化输入使某类别 logit 最大化：

```
x* = argmax (z_c(x) - λ||x||²)
```

观察结果：

* 不同类别具有不同频率/颜色分布
* 但缺乏明确语义结构

### 3️⃣ Error Analysis

* 模型依赖：

  * 颜色分布
  * 粗略几何结构
* 缺乏：

  * 空间局部建模能力

## 🧩 依赖环境

```bash
python >= 3.8
numpy
matplotlib
Pillow
scikit-learn
```

安装：

```bash
pip install numpy matplotlib pillow scikit-learn
```
