''' 
code from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py 
'''


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.pooling import DiffStride 


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_type='A', downsample_type='default'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            if shortcut_type == 'A':
                """
                For CIFAR10 ResNet paper uses shortcut_type A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif shortcut_type == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        self.downsample = nn.Identity() 
        if downsample_type == 'diffstride': 
            self.downsample = DiffStride() 

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.downsample(out)
        out = F.relu(out)

        out = self.bn2(self.conv2(out))
        out += self.downsample(self.shortcut(x))
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, shortcut_type='A', downsample_type='default'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, 
                                       shortcut_type=shortcut_type, downsample_type=downsample_type)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, 
                                       shortcut_type=shortcut_type, downsample_type=downsample_type)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, 
                                       shortcut_type=shortcut_type, downsample_type=downsample_type)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, shortcut_type, downsample_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, shortcut_type, downsample_type))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
