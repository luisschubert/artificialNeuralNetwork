# artificialNeuralNetwork – Computer Vision
> ## Recognizing Handwritten Digits (MNIST)

Implemented a Deep Neural Network for image classification.<br>
Images representing handwritten digits from 0 - 9.<br>
Network is trained using 60,000 images.<br>
Network classification performance is tested using 10,000 images.<br>


## Tuning the Learning Rate:
>Training Data: 60000 images<br>
Query Data: 10000 images<br>
Activation Function: Sigmoid<br>
Layers: (784, 100, 10)<br>

Sample Number | Learning Rate | Success Rate | Failure Rate
:--------------:|---------------|--------------|--------------
1 | 0.5 | 91.160 % | 8.840 %
2 | 0.25 | 94.010 % | 5.990 %
3 | 0.2 | 94.380 %| 5.620 %
4 | 0.1 | 94.620 % | 5.380 %
5 | 0.05 | 94.110 % | 5.890 % 
6 | 0.3 | 93.710 % | 6.290 %
7 | 0.6 | 90.180 % | 9.820 %
8 | 0.08 | 94.510 % | 5.490 %
9 | 0.15 | 94.590 % | 5.410 % 
10 | 0.125 | 95.020 % | 4.980 % 
11 | 0.13 | 94.570 % | 5.430 %
12 | 0.11 | 94.210 % | 5.790 %
13 | 0.12 | 94.560 % | 5.440 % 

Best Learning Rate | Success Rate | Failure Rate
---|---|---
0.125 | 95.020 % | 4.980 % 

## Data Augmentation Training Approach
Rotate the training images by 15° and -15° to triple the size of the training data.<br>

>Training Data: 180000 images<br>
Query Data: 10000 images<br>
Activation Function: Sigmoid<br>
Layers: (784, 100, 10)<br>

Sample Number | Learning Rate | Success Rate | Failure Rate
:--------------:|---------------|--------------|--------------
1 | 0.125 | 94.940 % | 5.060 %
2 | 0.15 | 94.940 % | 5.060 %
3 | 0.05 | 95.410 % | 4.590 %
4 | 0.03 | 94.960 % | 5.040 %
5 | 0.04 | 94.860 % | 5.140 % 
6 | 0.055 | 95.060 % | 4.940 %

Best Learning Rate | Success Rate | Failure Rate
---|---|---
0.05 | 95.410 % | 4.590 %

## Training with multiple Epochs

Sample Number | Learning Rate | Epochs | Success Rate | Failure Rate
:------------:|--------|--------------|--------------|--------------
1 | 0.5 | 10 | 93.280 % | 6.720 %
2 | 0.1 | 10 | 96.030 % | 3.970 %
3 | 0.15 | 10 |  96.440 % | 3.560 %
4 | 0.2 | 10 | 96.860 % | 3.140 %
5 | 0.3 | 10 | 97.060 % | 2.940 %
6 | 0.4 | 10 | 97.260 % | 2.740 %
7 | 0.5 | 10 | 91.590 % | 8.410 % // THIS IS WEIRD
8 | 0.5 | 10 | 92.310 % | 7.690 % // What is going on here ???
9 | 0.5 | 10 | 93.090 % | 6.910 %
10 | 0.35 | 10 | 93.390 % | 6.610 %
11 | 0.3 | 10 | 94.720 % | 5.280 %
12 | 0.25 | 10 | 95.580 % | 4.420 % 
13 | 0.3 | 10 | 94.070 % | 5.930 %
13 | 0.28 | 10 | 95.150 % | 4.850 %
14 | 0.32 | 10 | 93.860 % | 6.140 %
15 | 0.3 | 10 | 94.810 % | 5.190 %
16 | 0.3 | 2 | 94.040 % | 5.960 %
17 | 0.3 | 2 | 94.440 % | 5.560 %
18 | 0.3 | 2 | 94.520 % | 5.480 %
19 | 0.001 | 10 | 92.030 % | 7.970

## Discussion

The classification scores of networks with identical learning rates varied.
The only other thing that changes are how the weight matrices are initialized.

Next I should attempt to control the creation of the weight matrices and experiment with what works better.

There needs to be a better way to test these networks with multiple different paramters at once and find ones that seem to work better.


## Future Work
- [ ] visualize the weights with a bokeh heatmap
- [ ] implement an alternative train and query method of CIFAR 


