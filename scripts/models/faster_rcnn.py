import torchvision
from torchvision.models.detection import FasterRCNN as frcnn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .feature_extractor import MobileNetV2_pt


def FasterRCNN(num_classes):
    """Faster-RCNN architecture construction, whose backbone
    is a MobileNet. 
    Keyword arguments:
    - num_classes: number of classes     
    """
    backbone = MobileNetV2_pt()
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes = ((32,64,128,256,512),),
        aspect_ratios= ((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size = 7,
                                                    sampling_ratio = 2)
    model = frcnn(backbone,
                    num_classes = num_classes,
                    rpn_anchor_generator = anchor_generator,
                    box_roi_pool = roi_pooler)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
