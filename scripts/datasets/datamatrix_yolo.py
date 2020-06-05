import torch
import os
import numpy as np
import pandas as pd
import cv2
from utils.yolov3.yolo_utils import Compose, ImageBaseAug, ResizeImage, ToTensor
import albumentations as albu


PATH = "/home/tmr/data-matrix-dataset-loading/dataset_resized_4/"
PATH_IMAGES = os.path.join(PATH, "images")
PATH_LABELS = os.path.join(PATH, "labels")

class DataMatrixDataset(object):
    def __init__(self, img_size, mode = 'train',  is_debug=False):
        self.mode = mode
        self.img_size = img_size
        self.max_objects = 50
        self.is_debug = is_debug
        
        if self.mode == "train" or self.mode == "val":
            self.imgs = os.listdir(os.path.join(PATH_IMAGES, self.mode))
            self.lbls = os.path.join(PATH_LABELS, "data_matrix_" + self.mode + ".json")
            self.labels_file = pd.read_json(self.lbls)
            if self.mode == "val":
                idxs = [idx for idx in range(self.labels_file.shape[0])]
                self.labels_file.set_index([pd.Index(idxs)], inplace = True) 
        else:
            raise Exception("Oops. There are only two models: train and val!")
        self.transforms = Compose()
        if mode == 'train':
            self.transforms.add(ImageBaseAug())
        self.transforms.add(ResizeImage(self.img_size))
        self.transforms.add(ToTensor(self.max_objects, self.is_debug))
        
    def __getitem__(self, idx):
        filename = self.labels_file["External ID"][idx]
        img = cv2.imread(os.path.join(PATH_IMAGES, self.mode, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
        width, height = img.shape[1], img.shape[0]
        if height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            width, height = img.shape[1], img.shape[0]
        img = np.array(img)
        lbl_file_row = self.labels_file[self.labels_file.index == idx ]
        lbl = lbl_file_row["Label"][idx]
        num_objs = len(lbl['DataMatrix'])
        target = []
        for n_bbox in range(num_objs):
            pts = lbl['DataMatrix'][n_bbox]
            xmin = width
            ymin= height
            xmax = 0
            ymax = 0
            for i in range(4):
                if (pts['geometry'][i]['x']>xmax):
                    xmax = pts['geometry'][i]['x']
                if (pts['geometry'][i]['x']<xmin):
                    xmin = pts['geometry'][i]['x']
                if (pts['geometry'][i]['y']>ymax):
                    ymax = pts['geometry'][i]['y']
                if (pts['geometry'][i]['y']<ymin):
                    ymin = pts['geometry'][i]['y']
            coordinates = [xmin, ymin,xmax,ymax]
            normalized_bbox = albu.augmentations.bbox_utils.normalize_bbox(
                (coordinates), 
                height, 
                width)
            bbox_w = normalized_bbox[2] - normalized_bbox[0]
            bbox_h = normalized_bbox[3] - normalized_bbox[1]
            target.append(
                [1, 
                 normalized_bbox[0] + bbox_w / 2, 
                 normalized_bbox[1] + bbox_h / 2,
                 bbox_w,
                 bbox_h])
                        
        sample = {'image':img, 
                  'label':target
                 }
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = os.path.join(PATH_IMAGES,self.mode, filename)
        sample["origin_size"] = str([width, height])
        return sample
       
    
    def __len__(self):
        return self.labels_file.shape[0]     
             
