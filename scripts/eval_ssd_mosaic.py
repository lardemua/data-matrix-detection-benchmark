import torch
import sys
import numpy as np
import time 

from models.ssd.ssd import MobileNetV2SSD_Lite
from datasets.datamatrix import DataMatrixDataset
from utils.evaluation import convert_to_coco_api, CocoEvaluator
from utils.tools import get_arguments
from models.ssd.predictor import Predictor

def chop_img(img, min_percent_w = 0.016000000000000014, min_percent_h = 0.03216666666666662, sample_size = 1024):
    height, width = img.shape[0], img.shape[1]
    n_w = width / sample_size
    n_h = height / sample_size
    
    # How the image is chopped? n_horizontal_sample * n_vertical_samples 
    if (n_w - int(n_w)) > min_percent_w:
        n_horizontal_samples = int(n_w) + 1
    else:
        n_horizontal_samples = int(n_w)
    if (n_h - int(n_h)) > min_percent_h:
        n_vertical_samples = int(n_h) + 1
    else:
        n_vertical_samples = int(n_h)
    
    #Chopping:
    samples = []
    for v in range(n_vertical_samples ):
        for h in range(n_horizontal_samples ):
            if h == n_horizontal_samples - 1: # last chop horizontally
                initial_h = -sample_size
                final_h = width
            else:
                initial_h = h * sample_size
                final_h = (h + 1) * sample_size
            if v == n_vertical_samples - 1: 
                initial_v = -sample_size
                final_v = height 
            else:
                initial_v = v * sample_size
                final_v = (v + 1) * sample_size

            sample = img[initial_v : final_v, initial_h : final_h, ...]
            samples.append((sample, (h, v)))    
    return samples

def transform_bboxes(orig_shape, sample, last_sample, boxes, sample_size):
    for n_box in range(len(boxes)):
        if sample[1][0] == last_sample[1][0]: # last col pos
            boxes[n_box][0] = orig_shape[1] - (sample_size - boxes[n_box][0])
            boxes[n_box][2] = orig_shape[1] - (sample_size - boxes[n_box][2])
        else: 
            boxes[n_box][0] = boxes[n_box][0] + sample[1][0] * sample_size
            boxes[n_box][2] = boxes[n_box][2] + sample[1][0] * sample_size 

        if sample[1][1] == last_sample[1][1]: # last row pos
            boxes[n_box][1] = orig_shape[0] - (sample_size - boxes[n_box][1])
            boxes[n_box][3] =  orig_shape[0] - (sample_size - boxes[n_box][3])
        else:
            boxes[n_box][1] = boxes[n_box][1] + sample[1][1] * sample_size
            boxes[n_box][3] = boxes[n_box][3] + sample[1][1] * sample_size

    return boxes

def mosaic_result_ssd(initial_img, predictor, sample_size = 1024):
    samples = chop_img(initial_img, sample_size=sample_size)
    result = {}
    sample_i = 0
    init = time.time()
    while len(result) == 0 and sample_i < (samples[-1][1][0] + 1) * (samples[-1][1][1] + 1):
        boxes, labels, probs = predictor.predict(samples[sample_i][0], 10, 0.2)
        if labels.shape  != torch.Size([0]):
            boxes = transform_bboxes(initial_img.shape, samples[sample_i], samples[-1], boxes, sample_size)
            result["boxes"] = boxes 
            result["scores"] = probs
            result["labels"] = labels
        sample_i += 1
    for idx in range(sample_i,len(samples)):
        boxes, labels, probs = predictor.predict(samples[idx][0], 10, 0.2)
        if labels.shape != torch.Size([0]):
            boxes = transform_bboxes(initial_img.shape, samples[idx], samples[-1], boxes, sample_size)
            result["boxes"] = torch.cat((result["boxes"], boxes))
            result["scores"] = torch.cat((result["scores"], probs))
            result["labels"] = torch.cat((result["labels"], labels))
    proc_time = time.time() - init
            
    return result, proc_time


def evaluate(predictor, val_ds):
    """Evaluation of the validation set 
    Keyword arguments:
    - predictor: model after training with the respective weights
    - data_loader: validaton set in the loader format
    - device: device on which the network will be evaluated
    """
    coco = convert_to_coco_api(val_ds)
    coco_evaluator = CocoEvaluator(coco)
    evaluator_times = []
    proc_times = []
    for i in range(len(val_ds)):
        image, targets = val_ds[i]
        if image.shape[0] > 1080 and image.shape[1] > 1920:
            sample_size = 1024
        else:
            sample_size = 512
        results, time_fps = mosaic_result_ssd(image, predictor, sample_size = sample_size)
        proc_times.append(time_fps)
        res = {targets['image_id'].item(): results}
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
dev = args.device if args.device is not None else "cpu"
device = torch.device(dev)

if args.model == 'ssd512':
    from utils.ssd import ssd512_config as config
    net = MobileNetV2SSD_Lite(2, model = "ssd512",is_test = True, device = device)
else:
    sys.exit("You did not pick the right script! Exiting...")

if (args.dataset == 'datamatrix'):
  val_ds = DataMatrixDataset(mode = 'val')
  
  
# Loading weights to the network
if args.state_dict:
    net.load(args.state_dict)
    print('\nthe model was loaded successfully!')
else:
    raise ValueError("You have to load a model through the --state_dict argument!")

# Creating Predctor
candidate_size = 50 
sigma = 0.5 
predictor = Predictor(net,config.image_size, 
                     config.image_mean,
                     config.image_std,
                     iou_threshold = config.iou_threshold,
                     candidate_size = candidate_size,
                     sigma = sigma,
                     device = device)

evaluate(predictor, val_ds)