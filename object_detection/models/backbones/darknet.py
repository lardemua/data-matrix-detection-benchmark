import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



#YOLOV3 Darknet
def conv_block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 
                            kernel_size, stride, padding, bias = False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.LeakyReLU(0.1))
    ]))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(in_channels, out_channels // 2, kernel_size = 1, padding = 0)
        self.conv2 = conv_block(out_channels // 2, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return input + x
    
def make_layer(in_channels, out_channels, num_blocks, stride):
    layers = [
        conv_block(in_channels, out_channels, stride=stride),
    ]
    
    for _ in range(num_blocks):
        layers += [BasicBlock(out_channels, out_channels)]
    return nn.Sequential(*layers)

class DarkNet53(nn.Sequential):
    def __init__(self, in_channels = 3, num_classes = 1000):
        
        features = nn.Sequential(OrderedDict([
            ('stem', conv_block(in_channels, 32, stride=1)),
            ('layer1', make_layer(32, 64, 1, stride=2)),
            ('layer2', make_layer(64, 128, 2, stride=2)),
            ('layer3', make_layer(128, 256, 8, stride=2)),
            ('layer4', make_layer(256, 512, 8, stride=2)),
            ('layer5', make_layer(512, 1024, 4, stride=2)),
        ]))
        
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Maybe add a Dropout here!!!
            nn.Linear(1024, num_classes),
        )
        
        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))
        