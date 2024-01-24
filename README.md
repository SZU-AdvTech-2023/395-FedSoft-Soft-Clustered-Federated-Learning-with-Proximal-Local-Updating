# 软聚类联邦学习复现

复现文章 : FedSoft: Soft Clustered Federated Learning with Proximal Local Updating (AAAI 2022)

## 环境
复现实验在NVIDIA Tesla P100上运行，使用cuda版本为10.2.
主要的python环境如下:

* Python 3.6
* PyTorch 1.10.0+cu102
* Torchvision 0.11.0+cu102
* Numpy 1.18.5
* Scikit-learn 0.24.2

## 数据集

复现的FedSoft算法在基于Cifar10、MNIST、EMNIST和Digits5生成的7种数据集上的实验。

具体为：

* **Cifar-2set**：Cifar10和Cifar10旋转90度
* **MNIST-2set**：MNIST和MNIST 90°
* **MNIST-4set**：MNIST、MNIST 90°、MNIST 180°和MNIST 180°
* **Letters-2set**：EMNIST小写字母和EMNIST大写字母
* **Letters-4set**：EMNIST小写字母、EMNIST大写字母、EMNIST小写字母 90°和EMNIST大写字母 90°
* **Letters-8set**：EMNIST小写字母和EMNIST大写字母，并将其分别旋转90°、180°、270°
* **Digits5**：MNIST、MNIST-M、SYN、USPS和SVHN

## 运行

* python main_cifar_2set.py
* python main_mnist_2set.py
* python main_mnist_4set.py
* python main_letters_2set.py
* python main_letters_4set.py
* python main_letters_8set.py
* python main_digits5.py

## 引用

```
@inproceedings{ruan2022fedsoft,
  title={Fedsoft: Soft clustered federated learning with proximal local updating},
  author={Ruan, Yichen and Joe-Wong, Carlee},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={7},
  pages={8124--8131},
  year={2022}
}
```

