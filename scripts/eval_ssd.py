import torch
import sys
import numpy as np
import time 

from models.ssd.ssd import MobileNetV2SSD_Lite
from datasets.bdd100k import BDD100kDataset
from utils.evaluation import convert_to_coco_api, CocoEvaluator
from utils.tools import get_arguments
from models.ssd.predictor import Predictor




def evaluate(predictor, val_ds):
    """Evaluation of the validation set 
    Keyword arguments:
    - model: model after training with the respective weights
    - data_loader: validaton set in the loader format
    - device: device on which the network will be evaluated
    """
    coco = convert_to_coco_api(val_ds)
    coco_evaluator = CocoEvaluator(coco)
    evaluator_times = []
    for i in range(len(val_ds)):
        image, targets = val_ds.__getitem__(i)
        boxes, labels, probs = predictor.predict(image, 10 , 0.5)
        if boxes.size()[0] == 0:
            continue
        outputs = {'boxes': boxes,
                'labels':labels,
                'scores': probs}
        res = {targets['image_id'].item(): outputs}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_times.append(time.time() - evaluator_time) 
    
    print("Averaged stats:", np.mean(evaluator_times))
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize() 
    return coco_evaluator 


args = get_arguments()
device = torch.device("cpu")

if args.model == 'ssd512':
    from utils.ssd import ssd512_config as config
    net = MobileNetV2SSD_Lite(11, model = "ssd512",is_test = True, device = device)
else:
    sys.exit("You did not pick the right script! Exiting...")

if (args.dataset == 'bdd100k'):
  val_ds = BDD100kDataset(mode = 'val')
  
  
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
