import torch
import torch.nn as nn
use_cuda = torch.cuda.is_available()


class dw_conv(nn.Module):
    # Depthwise convolution, currently slow to train in PyTorch
    def __init__(self, in_dim, out_dim, stride):
        super(dw_conv, self).__init__()
        self.dw_conv_k3 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=stride, groups=in_dim, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_conv_k3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class point_conv(nn.Module):
    # Pointwise 1 x 1 convolution
    def __init__(self, in_dim, out_dim):
        super(point_conv, self).__init__()
        self.p_conv_k1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.p_conv_k1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNets(nn.Module):

    def __init__(self, num_classes, large_img):
        super(MobileNets, self).__init__()
        self.num_classes = num_classes
        if large_img:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                dw_conv(32, 32, 1),
                point_conv(32, 64),
                dw_conv(64, 64, 2),
                point_conv(64, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 128),
                dw_conv(128, 128, 2),
                point_conv(128, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 256),
                dw_conv(256, 256, 2),
                point_conv(256, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 2),
                point_conv(512, 1024),
                dw_conv(1024, 1024, 2),
                point_conv(1024, 1024),
                nn.AvgPool2d(7),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                dw_conv(32, 32, 1),
                point_conv(32, 64),
                dw_conv(64, 64, 1),
                point_conv(64, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 1024),
                dw_conv(1024, 1024, 1),
                point_conv(1024, 1024),
                nn.AvgPool2d(4),
            )

        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenet(num_classes, large_img, **kwargs):
    r"""PyTorch implementation of the MobileNets architecture
    <https://arxiv.org/abs/1704.04861>`_.
    Model has been designed to work on either ImageNet or CIFAR-10
    Args:
        num_classes (int): 1000 for ImageNet, 10 for CIFAR-10
        large_img (bool): True for ImageNet, False for CIFAR-10
    """
    model = MobileNets(num_classes, large_img, **kwargs)
    if use_cuda:
        model = model.cuda()
    return model
