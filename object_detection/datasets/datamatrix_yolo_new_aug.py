import torch
import os
import numpy as np
import pandas as pd
import cv2
from utils.ssd.transforms_ssd import TrainAugmentation
from utils.yolov3.config import training_params as config
import albumentations as albu


PATH = "/home/tmr/data-matrix-dataset-loading/dataset_resized_4/"
PATH_IMAGES = os.path.join(PATH, "images")
PATH_LABELS = os.path.join(PATH, "labels")

class DataMatrixDataset(object):
    def __init__(self, transformed = True, mode = 'train',  is_debug=False):
        self.mode = mode
        self.max_objects = 50
        self.is_debug = is_debug
        self.transforms = None
        self.transformed = transformed
        
        if self.mode == "train" or self.mode == "val":
            self.imgs = os.listdir(os.path.join(PATH_IMAGES, self.mode))
            self.lbls = os.path.join(PATH_LABELS, "data_matrix_" + self.mode + ".json")
            self.labels_file = pd.read_json(self.lbls)
            if self.mode == "val":
                idxs = [idx for idx in range(self.labels_file.shape[0])]
                self.labels_file.set_index([pd.Index(idxs)], inplace = True) 
        else:
            raise Exception("Oops. There are only two modes: train and val!")
        if (mode == 'train' and self.transformed):
            self.transforms = TrainAugmentation(config["input_shape"]["height"], np.array([127,127,127]), 128.0)
        
    def __getitem__(self, idx):
        filename = self.labels_file["External ID"][idx]
        img = cv2.imread(os.path.join(PATH_IMAGES, self.mode, filename))
        width, height = img.shape[1], img.shape[0]
        if height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            width, height = img.shape[1], img.shape[0]
        img = np.array(img)
        lbl_file_row = self.labels_file[self.labels_file.index == idx ]
        lbl = lbl_file_row["Label"][idx]
        num_objs = len(lbl['DataMatrix'])
        boxes = []
        obj_labels = []
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
            boxes.append([xmin, ymin,xmax,ymax])
            obj_labels.append(1)
        boxes = np.array(boxes, dtype = np.float32)
        labels = np.array(obj_labels, dtype = np.int64)    
        if self.transforms is not None:
            img, boxes, labels = self.transforms(img, boxes, labels)
            target = []
            for i in range(len(boxes)):
                bcx = (boxes[i][2] - boxes[i][0]) / 2
                bcy = (boxes[i][3] - boxes[i][1]) / 2
                bw = boxes[i][2] - boxes[i][0]
                bh = boxes[i][3] - boxes[i][1]
                target.append([labels[i], bcx,bcy,bw,bh])
            filled_labels = np.zeros((self.max_objects, 5), np.float32)
            filled_labels[:len(labels), :] = target
            sample = {'image':img, 
                      'label':torch.from_numpy(filled_labels)
                 }
        else:
            sample = {
                "image":img,
                "label":(labels, boxes) 
            }
        sample["image_path"] = os.path.join(PATH_IMAGES,self.mode, filename)
        sample["origin_size"] = str([width, height])
        return sample
 
       
    
    def __len__(self):
        return self.labels_file.shape[0]     
             
