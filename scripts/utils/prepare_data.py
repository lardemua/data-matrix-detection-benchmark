import albumentations as albu
from albumentations.pytorch import ToTensor

def get_tfms_faster():
  """Function that returns the transformations to
  be applied to both loaders (training, validation) 
  Keyword arguments:
  - cropsize: tupple
  - scalling: list of two values     
  """

  train_tfms = albu.Compose([
    albu.Resize(600,800),
    albu.HorizontalFlip(p = 1),
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
  targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
  return images, targets