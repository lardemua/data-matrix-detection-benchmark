import torch
from torch.nn import (Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d)
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from collections import namedtuple
import sys
from object_detection.models.backbones.mobilenetv2 import (MobileNetV2, InvertedResidual)
from object_detection.models.backbones.resnet50 import(ExtraResidual, 
                                                       Resnet50, 
                                                       Bottleneck) 
from object_detection.utils.ssd.ssd_utils import (convert_locations_to_boxes, 
                                                  center_form_to_corner_form)


GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])
class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x) # Extra layers results
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)



# ResNet50

def Resnet50SSD(num_classes, device, is_test=False):
    from object_detection.utils.ssd import ssd512_config_resnet as config 
    inp_size = config.image_size
    resnet50 = Resnet50(Bottleneck, [3, 4, 6, 3], inp_size)
    base_net = resnet50.features
    source_layer_indexes = [
        8,
        14,
        len(base_net),
    ]
    extra_layers = ModuleList([
        ExtraResidual(512, 256, 0.5), #input should be equal to resnet50.inchannel
        ExtraResidual(256,256, 0.5),
        ExtraResidual(256,256, 0.5),
        ExtraResidual(256,256, 0.5)
        ])
    arm_channels = [512, 1024, 512, 256, 256, 256, 256]

    num_anchors = [4, 6, 6, 6, 6, 4, 4]
    regression_list = []
    classification_list = []
    for i in range(len(arm_channels)):
        regression_list += [
                        nn.Conv2d(
                        arm_channels[i],
                        num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
        classification_list += [
                        nn.Conv2d(
                        arm_channels[i],
                        num_anchors[i] * num_classes,
                        kernel_size=3,
                        padding=1)
                ]
    regression_headers = ModuleList(regression_list)
    classification_headers = ModuleList(classification_list)
    return SSD(num_classes, 
               base_net, 
               source_layer_indexes,
               extra_layers, 
               classification_headers, 
               regression_headers, 
               is_test = is_test, 
               config = config, 
               device = device)


# MobileNetV2 Lite

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def MobileNetV2SSD_Lite(num_classes, device, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    from object_detection.utils.ssd import ssd512_config as config 
    n_last_chans = 4
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm,
                           onnx_compatible=onnx_compatible).features

    source_layer_indexes = [
        GraphPath(14, 'conv', 3),
        19,
    ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])


    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1280, out_channels = 6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels = 6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels = 6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels = n_last_chans * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels = n_last_chans * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels = 6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels = 6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels = 6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels = n_last_chans * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels = n_last_chans * num_classes, kernel_size=1),
    ])
   
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, device = device)


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)