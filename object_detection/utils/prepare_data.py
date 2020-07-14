import albumentations as albu
from albumentations.pytorch import ToTensor

def get_tfms_faster(ds):
  """Function that returns the transformations to
  be applied to both loaders (training, validation) 
  Keyword arguments:
  - cropsize: tupple
  - scalling: list of two values     
  """

  if ds == "datamatrix":
    train_tfms = albu.Compose([
      albu.OneOf([
              albu.augmentations.transforms.RandomSizedBBoxSafeCrop(480,640,p=0.2),
              albu.augmentations.transforms.RandomSizedBBoxSafeCrop(960,1280,p=0.2),
              albu.augmentations.transforms.Resize(750,1000, p=0.6),
          ], p=1),
      albu.augmentations.transforms.RandomBrightness(limit=0.5),
      albu.augmentations.transforms.RandomContrast(limit=0.5),
      albu.HorizontalFlip(),
      ToTensor(),
    ],bbox_params = albu.BboxParams(format='pascal_voc', 
                                  min_area = 0., 
                                  min_visibility = 0., 
                                  label_fields=['labels']))
    
  elif ds == "coco":
    train_tfms = albu.Compose([
      albu.augmentations.transforms.RandomBrightness(limit=0.5),
      albu.augmentations.transforms.RandomContrast(limit=0.5),
      albu.HorizontalFlip(),
      albu.VerticalFlip(),
      ToTensor(),
    ],bbox_params = albu.BboxParams(format='pascal_voc', 
                                  min_area = 0., 
                                  min_visibility = 0., 
                                  label_fields=['labels']))
    
  val_tfms = albu.Compose([
    ToTensor(),
  ],bbox_params = albu.BboxParams(format='pascal_voc', 
                                min_area = 0., 
                                min_visibility = 0., 
                                label_fields=['labels']))

  return train_tfms, val_tfms



def collate_fn(batch):
  """Function that organizes the batch
  Keyword arguments:
  - batch 
  """

  return list(zip(*batch))

def transform_inputs(images, targets, device):
  """Prepares data to pass forward the model
  during training 
  Keyword arguments:
  - images from batch
  - targets from batch
  - GPU device (e.g. "cuda:2")     
  """
  images = list(image.to(device) for image in images)
  targets = [{k: v.to(device) for k, v in t.items() } for t in targets]
  return images, targets