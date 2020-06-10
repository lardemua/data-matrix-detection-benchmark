import torch
import os
import numpy as np
import pandas as pd
import cv2


PATH = "/home/tmr/data-matrix-dataset-loading/dataset_resized_4/"
PATH_IMAGES = os.path.join(PATH, "images")
PATH_LABELS = os.path.join(PATH, "labels")

class DataMatrixDataset(object):
    def __init__(self, transforms = None, target_transform = None, mode = "train"):
        self. transforms = transforms 
        self.target_transform = target_transform
        self.mode = mode
        
        if self.mode == "train" or self.mode == "val":
            self.imgs = os.listdir(os.path.join(PATH_IMAGES, self.mode))
            self.lbls = os.path.join(PATH_LABELS, "data_matrix_" + self.mode + ".json")
            self.labels_file = pd.read_json(self.lbls)
            if self.mode == "val":
                idxs = [idx for idx in range(self.labels_file.shape[0])]
                self.labels_file.set_index([pd.Index(idxs)], inplace = True) 
        else:
            raise Exception("Oops. There are only two models: train and val!")
        
    def __getitem__(self, idx):
        filename = self.labels_file["External ID"][idx]
        #img = Image.open(os.path.join(PATH_IMAGES, self.mode, filename)).convert('RGB')
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
        boxes = []
        obj_labels = []
        for n_bbox in range(num_objs):
            coordinates = lbl['DataMatrix'][n_bbox]
            xmin = width
            ymin= height
            xmax = 0
            ymax = 0
            for i in range(4):
                if (coordinates['geometry'][i]['x']>xmax):
                    xmax = coordinates['geometry'][i]['x']
                if (coordinates['geometry'][i]['x']<xmin):
                    xmin = coordinates['geometry'][i]['x']
                if (coordinates['geometry'][i]['y']>ymax):
                    ymax = coordinates['geometry'][i]['y']
                if (coordinates['geometry'][i]['y']<ymin):
                    ymin = coordinates['geometry'][i]['y']
            boxes.append([xmin, ymin,xmax,ymax])
            obj_labels.append(1)
        boxes_ssd = np.array(boxes, dtype = np.float32)
        labels_ssd = np.array(obj_labels, dtype = np.int64)
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.ones((num_objs), dtype = torch.int64)
        image_id = torch.tensor(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        sample = {'image':img, 
                  'bboxes':boxes,
                  'labels':labels
                 }
        if (self.transforms is not None) and (self.target_transform is None):
            augmented = self.transforms(**sample)
            img = augmented['image']
            target['boxes'] = torch.as_tensor(augmented['bboxes'],  dtype = torch.float32)
            target['labels'] = torch.as_tensor(augmented['labels'], dtype = torch.int64)
            
        elif (self.transforms is not None) and (self.target_transform is not None): #SSD
            img, boxes, labels = self.transforms(img, boxes_ssd, labels_ssd)
            target['boxes'] = boxes
            target['labels'] = labels
            boxes, labels = self.target_transform(target['boxes'], target['labels'])
            return img, boxes, labels 
        
        return img, target
       
    
    def __len__(self):
        return self.labels_file.shape[0]     
             
