import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
import time 

from object_detection.models.yolo.yolo_darknet import Darknet
from object_detection.datasets.datamatrix_yolo import DataMatrixDataset
from object_detection.utils.evaluation import convert_to_coco_api, CocoEvaluator
from object_detection.utils.tools import get_arguments
from object_detection.utils.prepare_data import transform_inputs, collate_fn
from object_detection.utils.yolo.yolo_utils import *

# Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54, #1.0,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 205.76,  #64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20, #0.225,  # iou training threshold
       'lr0': 0.01, #0.001 # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,#-4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)


@torch.no_grad()
def evaluate(model,data_loader,device):
    """Evaluation of the validation set 
    Keyword arguments:
    - model: model after training with the respective weights
    - data_loader: validaton set in the loader format
    - device: device on which the network will be evaluated
    """
    cpu_device = torch.device("cpu")
    model.eval().to(device)
    coco = convert_to_coco_api(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco)
    evaluator_times = []
    proc_times = []
    for images, targets in data_loader:
        res = {}
        images, targets = transform_inputs(images, targets, device)
        images_eval = [image.float()/255 for image in images] #normalized
        batch = torch.stack(images_eval)

        
        init = time.time()
        inf_out, eval_out = model(batch)
        proc_times.append(time.time() - init)
        
        output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres = 0.6)
        for si, pred in enumerate(output):
            height, width = images[si].shape[1:]
            if pred is None:
                box = torch.tensor([[0,0,0,0]])
                res.update({targets[si]["image_id"].item():
                            {"boxes": box,
                             "labels": torch.tensor([1]),
                             "scores" : torch.tensor([0])}})
            else:
                clip_coords(pred, (height, width))
                box = pred[:, :4].clone() 
                res.update({targets[si]["image_id"].item():
                           {"boxes":box,
                            "labels":pred[:, 5],
                            "scores":pred[:,4]
                           }})
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_times.append(time.time() - evaluator_time) 
    print("Averaged stats:", np.mean(evaluator_times))
    print("Averaged proc time:", np.mean(proc_times))
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


args = get_arguments()

if (args.model == "yolov3" or args.model == "yolov3_spp" or args.model == "yolov4"):
    if args.model == "yolov3":
        yolo_config = "object_detection/utils/yolo/yolov3.cfg"
    elif args.model == "yolov3_spp":
        yolo_config = "object_detection/utils/yolo/yolov3_spp.cfg"
    else: #yolov4
        yolo_config = "object_detection/utils/yolo/yolov4.cfg"
        
    model = Darknet(yolo_config)
else:
    sys.exit("You did not pick the right script! Exiting...")
  
device = torch.device("cuda")
  
if args.dataset == 'datamatrix':    
    val_ds = DataMatrixDataset(mode = "test",
                                img_size = 896,
                                batch_size = args.batch_size,
                                hyp = hyp,
                                rect = True)
    
    val_loader = DataLoader(val_ds,
                        batch_size = args.batch_size,
                        num_workers = args.workers,
                        pin_memory = True,
                        collate_fn = collate_fn)
    
if args.state_dict:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print('\nthe model was loaded successfully!')
else:
    raise ValueError("You have to load a model through the --state_dict argument!")

evaluate(model, val_loader, device)