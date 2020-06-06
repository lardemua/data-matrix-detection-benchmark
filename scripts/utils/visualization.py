import cv2
import matplotlib.pyplot as plt
import numpy as np


LBLS_MAP = {
    0 : 'background',
    1 : 'data matrix',
} 
        
COLORMAP = {
    'BACKGROUND' : (30,30,30),
    'data matrix' : (255,0,0),
}

TEXT_COLOR = (255, 255, 255)


class VisualizationImg:
    """Visualization of img data and the respectives bounding boxes 
    Keyword arguments:
    - tuple (img, target). Both are torch tensors and the target has to have two fields: boxes
    (bounding boxes in the format xmin, ymin, xmax, ymax) adn the respective labels
    Example of usage for the Faster RCNN data loader:
    from datasets.bdd_dataset import BDD100kDataset
    from utils.visualization import VisualizationImg
    ds = BDD100kDataset()
    example = ds[1]
    vis = VisualizationImg(*example)
    img = vis.get_img()
    plt.imshow(img);  
    """
    def __init__(self, img,target, eval_frames = False, thresh = 0.5):
        self.img = img
        self.target = target
        self.eval_frames = eval_frames
        self.threshold = thresh

    def visualize_bbox(self, img, bbox, label, thickness=4):
        x_min, y_min, x_max, y_max = bbox
        class_name = LBLS_MAP[label]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORMAP[class_name], thickness) #top left and bottom right
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(img, (int(x_min), int(y_min) - int(1.3 * text_height)), (int(x_min) + text_width, int(y_min)), COLORMAP[class_name], -1)
        cv2.putText(img, class_name, (int(x_min), int(y_min) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def get_img(self):
        img_final = np.array(self.img, dtype = np.uint8).transpose(1,2,0)
        target = self.target
        boxes = []
        labels = []
        for i in range(len(target['boxes'])):
            if self.eval_frames == True and target['scores'][i]>self.threshold:
                boxes.append(list(np.array(target['boxes'][i])))
                labels.append(int(target['labels'][i]))
            elif self.eval_frames == False:
                boxes.append(list(np.array(target['boxes'][i])))
                labels.append(int(target['labels'][i]))
        for i in range(len(boxes)):
            img_final = self.visualize_bbox(img_final, boxes[i], labels[i])
        return img_final


