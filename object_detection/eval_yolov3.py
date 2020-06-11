import torch
import sys
import numpy as np
import time
import cv2

from models.yolov3.yolov3 import yolov3_darknet
from utils.yolov3.config import training_params as config
from utils.yolov3.yolo_utils import non_max_suppression, YOLOLoss
from datasets.datamatrix import DataMatrixDataset
from utils.evaluation import convert_to_coco_api, CocoEvaluator
from utils.tools import get_arguments

def preprocess_frame(frame):
    img = cv2.resize(frame,(config["input_shape"]["width"], config["input_shape"]["height"]))
    img = img.astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img

def evaluate(net, val_ds, device):
    """Evaluation of the validation set 
    Keyword arguments:
    - model: model after training with the respective weights
    - data_loader: validaton set in the loader format
    - device: device on which the network will be evaluated
    """
    coco = convert_to_coco_api(val_ds)
    coco_evaluator = CocoEvaluator(coco)
    evaluator_times = []
    processing_times = []
    it = iter(val_ds)
    yolo_losses = []
    for l in range(3):
        yolo_losses.append(YOLOLoss(config["priors_info"]["anchors"][l],
                                config["priors_info"]["classes"], (config["input_shape"]["width"], config["input_shape"]["height"])))
        
    for i in range(len(val_ds)):
        sample = next(it)
        img = sample[0]
        target = sample[1]
        img_proc = preprocess_frame(img)
        init_time = time.time()
        with torch.no_grad():
            outputs = net(img_proc.to(device))
        processing_times.append(time.time() - init_time)
        output_list = []
        for ll in range(3):
            output_list.append(yolo_losses[ll](outputs[ll]))
        output = torch.cat(output_list, 1)
        output = non_max_suppression(output, config["priors_info"]["classes"], conf_thres=0.01, nms_thres=0.45)
        ori_h, ori_w = img.shape[0], img.shape[1]
        pre_h, pre_w  = (config["input_shape"]["height"], config["input_shape"]["width"])
        if output[0] is not None:
            bboxes = []
            labels = []
            probs = []    
            for idx in range(len(output[0])):
                y1 = (output[0][idx][1] / pre_h) * ori_h
                x1 = (output[0][idx][0] / pre_w) * ori_w
                y2 = y1 + ((output[0][idx][3] - output[0][idx][1]) / pre_h) * ori_h
                x2 = x1 + ((output[0][idx][2] - output[0][idx][0]) / pre_w) * ori_w
                bboxes.append([x1,y1,x2,y2])
                labels.append(output[0][idx][-1])
                probs.append(output[0][idx][4])
            result = {}
            result["boxes"] = torch.tensor(bboxes)
            result["scores"] = torch.tensor(probs)
            result["labels"] = torch.tensor(labels)
            res = {target['image_id'].item(): result}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_times.append(time.time() - evaluator_time) 
        else:
            continue
            
    print("Averaged stats:", np.mean(evaluator_times))
    print("Averaged proc time:", np.mean(processing_times))
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize() 
    return coco_evaluator 

args = get_arguments()
device = torch.device(args.device)

if args.model == 'yolov3':
    net = yolov3_darknet(config, is_training = False)
    net.to(device)

else:
    sys.exit("You did not pick the right script! Exiting...")

if (args.dataset == 'datamatrix'):
    val_ds = DataMatrixDataset(mode = 'val')

# Loading weights to the network
if args.state_dict:
    state_dict = torch.load(args.state_dict, map_location = "cpu")
    net.load_state_dict(state_dict)
    print('\nthe model was loaded successfully!')
else:
    raise ValueError("You have to load a model through the --state_dict argument!")

evaluate(net, val_ds, device)
