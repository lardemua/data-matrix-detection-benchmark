import torch
import torch.optim as optim
import sys
import itertools
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from time import localtime, strftime

from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import (global_step_from_engine, ModelCheckpoint)


#object_detection modules
from object_detection.datasets.datamatrix import DataMatrixDataset
from object_detection.models.ssd.ssd import (MobileNetV2SSD_Lite, Resnet50SSD)
from object_detection.utils.tools import (get_arguments, get_scheduler)
from object_detection.utils.ssd.ssd_utils import  MatchPrior, freeze_net_layers
from object_detection.utils.ssd.transforms_ssd import (TrainAugmentation, TestTransform)
from object_detection.engine import (create_detection_trainer, create_detection_evaluator)
from object_detection.utils.evaluation import convert_to_coco_api
from object_detection.losses.multibox import MultiboxLoss


args = get_arguments()

if args.distributed:
    dist.init_process_group("nccl", init_method = "env://")
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    local_rank = args.local_rank
else:
    local_rank = 0

torch.cuda.set_device(local_rank)
device = torch.device("cuda")


if  (args.model == "ssd512") and (args.feature_extractor == "mobilenetv2"):
    from object_detection.utils.ssd import ssd512_config as config
    model = MobileNetV2SSD_Lite(2, device)
elif (args.model == "ssd512") and (args.feature_extractor == 'resnet50'):
    from object_detection.utils.ssd import ssd512_config_resnet as config
    model = Resnet50SSD(2, device)
else:
    sys.exit("You did not pick the right script! Exiting...")


target_transform = MatchPrior(config.priors,
                              config.center_variance,
                              config.size_variance,
                              0.5)

train_tfms = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
test_tfms = TestTransform(config.image_size, config.image_mean, config.image_std)


if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    params = model.parameters()

elif args.pretrained_model is not None:
    model.init_from_pretrained_ssd(args.pretrained_model)
    freeze_net_layers(model.base_net)
    params = itertools.chain(model.source_layer_add_ons.parameters(), 
                             model.extras.parameters(),
                             model.regression_headers.parameters(), 
                             model.classification_headers.parameters())
    params = [
        {'params': itertools.chain(
            model.source_layer_add_ons.parameters(),
            model.extras.parameters()
        ), 'lr': args.learning_rate},
        {'params': itertools.chain(
            model.regression_headers.parameters(),
            model.classification_headers.parameters()
        )}
    ]
else:
    params = model.parameters() # from scratch
    

model.to(device)

if (args.dataset == 'datamatrix'):
  train_ds = DataMatrixDataset(transforms = train_tfms, target_transform = target_transform)
  val_ds = DataMatrixDataset(mode = 'val')

if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_ds, **kwargs)
    kwargs['shuffle'] = False
    val_sampler = DistributedSampler(val_ds, **kwargs)
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=not args.distributed,
    drop_last=True,
    num_workers=args.workers,
)



coco_api_val_dataset = convert_to_coco_api(val_ds)

optimizer = torch.optim.AdamW(params,
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,)

scheduler = get_scheduler(optimizer,args.epochs, args.learning_rate, len(train_loader))

loss_fn = MultiboxLoss(config.priors, 
                         iou_threshold=0.5, 
                         neg_pos_ratio=3, 
                         center_variance=0.1, 
                         size_variance=0.2, 
                         device = device)



if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)

evaluator = create_detection_evaluator(args.model,
                                       model, 
                                       device, 
                                       coco_api_val_dataset,
                                       logging = local_rank == 0)

trainer = create_detection_trainer(args.model, 
                                   model, 
                                   optimizer, 
                                   device,
                                   val_ds,
                                   evaluator,
                                   loss_fn = loss_fn,
                                   logging = local_rank == 0
                                   )

trainer.add_event_handler(
    Events.ITERATION_COMPLETED, scheduler,
)

if local_rank == 0:
    dirname = strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())
    dirname = "checkpoints/" + args.dataset + "/" + args.feature_extractor + args.model + "/{}".format(dirname)
    
    checkpointer = ModelCheckpoint(
        dirname=dirname,
        filename_prefix=args.model,
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

trainer.run(train_loader, max_epochs=args.epochs)


