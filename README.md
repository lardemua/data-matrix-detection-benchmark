# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks using Deep Learning architectures.


# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Hardware](#hardware) 
- [Object Detection](#object-detection)
  * [Faster RCNN](#faster-rcnn)
  * [SSD512](#ssd512)
  * [YOLOV3](#yolov3)


# Overview
This repo is an aggregation of Faster RCNN, SSD512 and YOLOV3(608) for Data Matrix detection. The performance comparison between all models is also performed. The training set images size from the original dataset were 8000*6000. The original training set images were reduced 4 times due to the faster forward propagation of the single shot methods and the way how Python deals with memory deallocation (see [Python Garbage Collection](https://www.digi.com/resources/documentation/digidocs/90001537/references/r_python_garbage_coll.htm)).   

# Hardware

Training and Evaluation: Nvidia RTX2080ti

# Object Detection

## Faster RCNN

To train:
````
python train_faster.py -b 4 -lr 1e-4 -eps 25
````

To evaluate:
````
python eval_faster.py -sd <model.pth>
````

Evaluation (original dataset):  

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.312    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.642    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.272    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.006    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.192    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.486    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.179    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.364    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.365    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.078    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.270    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.527    |


Evaluation (resized dataset):

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.263    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.545    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.200    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.154    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.424    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.161    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.309    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.310    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.007    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.222    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.471    |


frame rate: 30 fps

## SSD512


To train:
````
python train_ssd.py -m ssd512 - 16 -lr 1e-3 -wd 4e-5 -eps 150
````

To evaluate:
````
python eval_faster.py -m ssd512 -sd <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.218    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.281    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.273    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.276    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.199    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.258    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.258    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.327    |

framerate: 100fps

## YOLOV3

To train:
````
python train_yolov3.py -m yolov3 -b 8 -wd 4e-5 -eps 200
````

To evaluate:
````
python eval_yolov3.py -m yolov3 -sd <model.pth>
````