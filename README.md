# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks.


# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Hardware](#hardware) 
- [Object Detection](#object-detection)
  * [Faster RCNN](#faster-rcnn)
  * [SSD512](#ssd512)
  * [YoloV3](#yolov3)


# Overview
This repo is an aggregation of Faster RCNN, SSD512 and YoloV3(608) for Data Matrix detection. The performance comparison between all models is performed at the end.

# Hardware

Training and Evaluation: Nvidia RTX2080ti

# Object Detection

## Faster RCNN

Evaluation:  

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

frame rate: 30 fps

## SSD512

Evaluation:

  |       Metric             |  IoU Thresholds |  Scales  |  maxDets  | AP/AR values |
  | :----------------------: | :-------------: | :------: | :-------: | :----------: |
  | Average Precision  (AP)  |     0.50:0.95   |     all  |    100    |     0.147    |
  | Average Precision  (AP)  |     0.50        |     all  |    100    |     0.221    |
  | Average Precision  (AP)  |     0.75        |     all  |    100    |     0.164    |
  | Average Precision  (AP)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Precision  (AP)  |     0.50:0.95   |   large  |    100    |     0.209    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |      1    |     0.148    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |     10    |     0.183    |
  | Average Recall     (AR)  |     0.50:0.95   |     all  |    100    |     0.183    |
  | Average Recall     (AR)  |     0.50:0.95   |   small  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |  medium  |    100    |     0.000    |
  | Average Recall     (AR)  |     0.50:0.95   |   large  |    100    |     0.262    |

framerate: 100fps

## YOLOV3 (608) 