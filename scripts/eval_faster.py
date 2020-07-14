import torch
from torch.utils.data import DataLoader
import time
import numpy as np

from object_detection.models.faster.faster_rcnn import (resnet50fpn_fasterRCNN, 
                                                        resnet50_fasterRCNN, 
                                                        mobilenetv2_fasterRCNN)
from object_detection.datasets.datamatrix import DataMatrixDataset 
from object_detection.utils.evaluation import convert_to_coco_api, CocoEvaluator
from object_detection.utils.prepare_data import get_tfms_faster,transform_inputs, collate_fn
from object_detection.utils.tools import get_arguments

@torch.no_grad()
def evaluate(model,data_loader,device):
    """Evaluation of the validation set 
    Keyword arguments:
    - model: model after training with the respective weights
    - data_loader: validaton set in the loader format
    - device: device on which the network will be evaluated
    """
    cpu_device = torch.device("cpu")
    model.eval()
    coco = convert_to_coco_api(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco)
    evaluator_times = []
    proc_times = []
    for image, targets in data_loader:
        image, targets = transform_inputs(image, targets, device)
        init = time.time()
        outputs = model(image)
        proc_times.append(time.time() - init)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
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

if args.model == 'faster':
    if (args.feature_extractor == 'mobilenetv2'):
        model = mobilenetv2_fasterRCNN(2)
    elif (args.feature_extractor == 'resnet50fpn'):
        model = resnet50fpn_fasterRCNN(2)
    elif (args.feature_extractor == 'resnet50'):
        model = resnet50_fasterRCNN(2)
else:
    sys.exit("You did not pick the right script! Exiting...")

train_tfms, val_tfms = get_tfms_faster(ds = args.dataset)
if args.dataset == 'datamatrix':
  val_ds = DataMatrixDataset(transforms = val_tfms, mode = 'test')
  val_loader = DataLoader(
     val_ds,
     batch_size = args.batch_size,
     shuffle=False,
     drop_last=False,
     num_workers=args.workers,
    collate_fn = collate_fn
    )

if args.state_dict:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    print('\nthe model was loaded successfully!')
else:
    raise ValueError("You have to load a model through the --state_dict argument!")

model.to(args.device)
evaluate(model, val_loader, args.device)



