import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


# pretrained torchvision 
def resnet50_pt(pretrained):
    """Resnet50 feature extractor to the Faster RCNN acrhitecture
    """
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    modules = list(resnet.children())[:-2] # delete the last fc layer and avg pool layers.
    features = nn.Sequential(*modules)
    return features
    
    
# from scratch    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ExtraResidual(nn.Module):
    def __init__(self, inp, oup, expand_ratio=1):
        super(ExtraResidual, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        
        self.use_res_connect = inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, kernel_size = 1, stride = 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(hidden_dim, oup, kernel_size = 3, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(oup),
        )
    
    def forward(self, x):
        # if self.use_res_connect:
        #     return x + self.conv(x)
        # else:
        return self.conv(x)                

class Resnet50(nn.Module):
    def __init__(self, block, num_blocks, size):
        super(Resnet50, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.ReLU(nn.BatchNorm2d(64))
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        # Bottom-up layers
        self.out_channels = [64,128,256,512]
        self.features = [nn.Sequential(self.conv1, self.bn1, self.max_pool1)]
        for i in range(len(self.out_channels)):
            if i == 0:
                self.features.extend([*self._make_layer(block, self.out_channels[i], num_blocks[i], stride=1)])
            else:
                self.features.extend([*self._make_layer(block, self.out_channels[i], num_blocks[i], stride=2)]) 
        self.inchannel = block.expansion * 512

        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.features.append(nn.Sequential(self.smooth1))
        self.features = nn.Sequential(*self.features)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Bottom-up
        x = self.features(x)
        return x