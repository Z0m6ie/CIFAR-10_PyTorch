# CIFAR10-PyTorch with MobileNets

### Basis
A generic implementation to train, validate &amp; test various models on the CIFAR 10 dataset. All in PyTorch.

currently the only model implemented is MobileNets, The implementation is based on my understanding of the original paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Howard et al.

https://arxiv.org/abs/1704.04861

Currently in PyTorch there is an issue with how Depthwise convolutions are handled and training is quite slow. This has been fixed on the master version of PyTorch but not yet on the standard version:

https://github.com/pytorch/pytorch/issues/1708

### Usage
Args:

```
num_classes (int): 1000 for ImageNet, 10 for CIFAR-10

large_img (bool): True for ImageNet, False for CIFAR-10
```

e.g. to call model for use on ImageNet:

`model = mobilenet(num_classes=1000, large_img=True)`

For use on CIFAR-10 call:

`model = mobilenet(num_classes=10, large_img=False)`

### Results

Please see the `CIFAR10-testbed.ipynb` as an example use case as shown below the CIFAR-10 version of the MobileNets architecture achieves `85.65%` on the test set after 50 epochs.

[Accuracy]: ./result.jpg "Accuracy"


![alt text][Accuracy]

### TensorboardX
Fantastic logger for tensorboard and pytorch, https://github.com/lanpa/tensorboard-pytorch

run tensorboard by opening a new terminal and run `"tensorboard --logdir runs"`
open tensorboard at http://localhost:6006/
