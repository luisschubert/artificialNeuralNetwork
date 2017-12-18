# artificialNeuralNetwork – Computer Vision
> ## Recognizing Handwritten Digits (MNIST)

Implemented a Deep Neural Network for image classification.<br>
Images representing handwritten digits from 0 - 9.<br>
Network is trained using 60,000 images.<br>
Network classification performance is tested using 10,000 images.<br>

### Todo:
- [ ] add introduction
- [ ] add sources
- [ ] gradient descent formula
- [ ] make more generic for n-hidden layers architecture
- [ ] show graphic to illustrate architecture
- [ ] show neural network vision visualization



### Tuning the Learning Rate:
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
