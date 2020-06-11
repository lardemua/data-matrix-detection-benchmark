
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


# FASTER RCNN fearure extractors
def resnet50_pt():
    """Resnet50 feature extractor to the Faster RCNN acrhitecture
    """
    resnet = torchvision.models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-2] # delete the last fc layer and avg pool layers.
    features = nn.Sequential(*modules)

    return features

def mobilenetv2_pt():
    """MobileNetV2 feature extractor to the Faster RCNN architecture
    """
    return torchvision.models.mobilenet_v2(pretrained=True).features
#---------------------------------------------------------------------------------------------------------------------

# SSD Feature Extractors

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.
def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 to the SSD acrhitecture 
    """
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
# Resnet50
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


#---------------------------------------------------------------------------------------------------------------------

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
        