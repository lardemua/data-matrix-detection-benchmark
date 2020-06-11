import copy
import torch
from ignite.engine import (Engine, 
                           Events, 
                           _prepare_batch, 
                           create_supervised_evaluator)
from ignite.metrics import RunningAverage, Loss
from object_detection.utils.prepare_data import transform_inputs
from ignite.contrib.handlers import ProgressBar
from object_detection.utils.evaluation import CocoEvaluator


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
        ProgressBar(persist=True) \
            .attach(trainer, ['loss', 'lr'])

    return trainer



def create_detection_evaluator(model, device, coco_api_val_dataset):
    def update_model(engine, batch):
        images, targets = batch
        images, targets = transform_inputs(images, targets, device)
        images_model = copy.deepcopy(images)

        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(images_model)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        engine.state.coco_evaluator.update(res)

        images_model = outputs = None

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




    

                


