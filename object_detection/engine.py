from ignite.engine import Engine, Events, _prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss
from object_detection.utils.prepare_data import transform_inputs
from ignite.contrib.handlers import ProgressBar

__all__ = [
    "create_detection_trainer",
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
        


def create_detection_trainer(model_name, model, optimizer, device, loss_fn = None, logging = True):
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

    if logging:
        ProgressBar(persist=True) \
            .attach(trainer, ['loss', 'lr'])

    return trainer