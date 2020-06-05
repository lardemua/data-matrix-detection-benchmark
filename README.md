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
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.307    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.624    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.260    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.001    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.188    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.485    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.179    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.361    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.362    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.023    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.270    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.535    |


Evaluation (resized dataset):

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.268    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.564    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.221    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.001    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.146    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.440    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.159    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.315    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.317    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.020    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.220    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.484    |


frame rate: 30 fps

## SSD512


To train:
````
python train_ssd.py -m ssd512 -b 16 -lr 1e-3 -wd 4e-5 -eps 150
````

To evaluate:
````
python eval_ssd.py -m ssd512 -sd <model.pth>
````

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.199    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.289    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.244    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.250    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.181    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.234    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.234    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.298    |

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

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.007    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.034    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.002    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.001    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.015    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.026    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.035    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.035    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.003    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.066    |