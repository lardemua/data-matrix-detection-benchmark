import copy
import torch
from ignite.engine import (Engine, 
                           Events, 
                           _prepare_batch, 
                           create_supervised_evaluator)
from ignite.metrics import RunningAverage, Loss
from ignite.contrib.handlers import ProgressBar
from object_detection.utils.prepare_data import transform_inputs
from object_detection.utils.evaluation import CocoEvaluator
from object_detection.models.ssd.predictor import Predictor
from object_detection.datasets.datamatrix_yolo import xyxy2xywh
from object_detection.utils.yolo.yolo_utils import *



__all__ = [
    "create_detection_trainer",
    "create_detection_evaluator"
]

def train_data(model_name, model, batch, loss_fn, device):
    if model_name == "faster":
        images, targets = batch
        images, targets = transform_inputs(images, targets, device)
        
        losses = model(images, targets)
        loss = sum([loss for loss in losses.values()])
    
    elif model_name == "ssd512":
        images, boxes, labels = batch
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        confidence, locations = model(images)

        regression_loss, classification_loss = loss_fn(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
    
    elif model_name == "yolov3":
        images, targets, paths, _ = batch
        images = images.to(device).float() / 255.0
        targets = targets.to(device)
        
        predictions = model(images)
        
        loss, loss_items = loss_fn(predictions, targets, model)
        loss *= len(batch) / 64 
    return loss
        


def create_detection_trainer(model_name, 
                             model, 
                             optimizer, 
                             device, 
                             val_loader,
                             evaluator, 
                             loss_fn = None, 
                             logging = True):
    def update_fn(_trainer, batch):
        """Training function
        Keyword arguments:
        - each bach 
        """
        model.train()
        optimizer.zero_grad()
        loss = train_data(model_name, model, batch, loss_fn, device)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x) \
    .attach(trainer, 'loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_optimizer_params(engine):
        param_groups = optimizer.param_groups[0]
        for h in ['lr', 'momentum', 'weight_decay']:
            if h in param_groups.keys():
                engine.state.metrics[h] = param_groups[h]
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        evaluator.run(val_loader)

    if logging:
        ProgressBar(persist=False) \
            .attach(trainer, ['loss', 'lr'])

    return trainer


def test_data(model_name, model, batch, device):
    if model_name == "faster":
        images, targets = batch
        images, targets = transform_inputs(images, targets, device)
        images_model = copy.deepcopy(images)
        
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(images_model)
        
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    
    
    elif model_name == "ssd512":
        from object_detection.utils.ssd import ssd512_config as config
        images, targets = batch    
        images_model = copy.deepcopy(images)
        candidate_size = 50 
        sigma = 0.5 
        predictor = Predictor(model,
                              config.image_size, 
                              config.image_mean,
                              config.image_std,
                              iou_threshold = config.iou_threshold,
                              candidate_size = candidate_size,
                              sigma = sigma,
                              device = device)
        boxes, labels, probs = predictor.predict(images_model, 10 , 0.2)
        if boxes.size()[0] == 0:
            outputs = {"boxes": torch.tensor([[0,0,0,0]]),
                       "labels": torch.tensor([0]),
                       "scores" : torch.tensor([0])}
        else:
            outputs = {"boxes": boxes,
                       "labels" : labels,
                       "scores": probs}
            
        res = {targets['image_id'].item(): outputs}
        
    elif model_name == "yolov3":
        images, targets = batch
        images, targets = transform_inputs(images, targets, device)
        images_model = copy.deepcopy(images)
        images_model = [image.float()/255 for image in images_model]
        batch_imgs = torch.stack(images_model)

        boxes = []
        labels_ = []
        scores = []
        res = {}
        nb, _, width, height = batch_imgs.shape
        whwh = torch.Tensor([width, height, width, height]).to(device)
        
        torch.cuda.synchronize()
        with torch.no_grad():
            inf_out, train_out = model(batch_imgs)
        output = non_max_suppression(inf_out, conf_thres=0.25, iou_thres = 0.6)
        for si, pred in enumerate(output):
            labels = targets[si]["boxes"]
            shapes = targets[si]["shapes"]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            
            if pred is None:
                outputs = {"boxes": torch.tensor([[0,0,0,0]]),
                           "labels": torch.tensor([1]),
                           "scores" : torch.tensor([0])}
            else:
                pred = clip_coords(pred, (height, width))
                box = pred[:, :4].clone()  # xyxy
                box = scale_coords(batch_imgs[si].shape[1:], box, shapes[0], shapes[1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                box = xywh2xyxy(box)
                for p in pred.tolist():
                    labels_.append([p[5]])
                    scores.append(round(p[4], 5))
                            
                outputs = {"boxes": box,
                           "labels": torch.tensor(labels_),
                           "scores": torch.tensor(scores)}
            res.update({targets[si]["image_id"].item(): outputs})
                
    images_model = outputs = None
    return images, targets, res 
        
    

def create_detection_evaluator(model_name, model, device, coco_api_val_dataset):
    def update_model(engine, batch):
        images, targets, res = test_data(model_name, model, batch, device)
        engine.state.coco_evaluator.update(res)
        return images, targets, res
    
    evaluator = Engine(update_model)
    ProgressBar(persist=False) \
        .attach(evaluator)
    
    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        model.eval()
        engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset)
            
    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        engine.state.coco_evaluator.synchronize_between_processes()
        print("\nResults val set:")
        engine.state.coco_evaluator.accumulate()
        engine.state.coco_evaluator.summarize()
     
        
        
    return evaluator






