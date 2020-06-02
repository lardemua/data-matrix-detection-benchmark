# data-matrix-detection-benchmark
Benchmark for the detection of data matrix landmarks. 

First result Faster-RCNN:

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