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



### Sample output:
>Learning Rate: 0.5<br>
Training Data: 60000 images<br>
Query Data: 10000 images<br>
Batches: 1<br>
Activation Function: Sigmoid<br>
Layers: (784, 100, 10)<br>
```
I should be seeing a 9
I think I'm seeing a 9
[ 0.00313555  0.00146607  0.00263237  0.00150256  0.01233606  0.00360587
  0.00315127  0.01133369  0.00347867  0.99205022]
------------------------

I should be seeing a 0
I think I'm seeing a 8
[ 0.09332549  0.05505109  0.23605739  0.04253107  0.01073366  0.03620878
  0.06260703  0.00313015  0.48662018  0.09641329]
------------------------

I should be seeing a 1
I think I'm seeing a 1
[  6.57109229e-05   9.98472400e-01   3.21613960e-02   8.14837740e-04
   1.73067710e-02   1.05666197e-03   6.16614259e-05   6.09729237e-03
   2.06032987e-04   3.23828917e-03]
------------------------

I should be seeing a 2
I think I'm seeing a 2
[  2.23093187e-04   1.87259876e-02   9.94259352e-01   1.97280621e-02
   3.23320103e-03   5.19855522e-03   1.54566502e-05   1.16371014e-03
   1.79712892e-03   1.44905780e-03]
------------------------

I should be seeing a 3
I think I'm seeing a 3
[  3.50486456e-04   2.16513299e-02   6.94023795e-03   9.96409716e-01
   7.87548483e-05   2.34399165e-03   8.63952060e-05   4.55877007e-04
   9.34328888e-05   5.68825660e-04]
------------------------

I should be seeing a 4
I think I'm seeing a 9
[  6.92391547e-03   2.99437980e-02   1.73178743e-03   8.10612962e-04
   6.63802694e-01   1.19922327e-02   3.89179389e-04   4.36548776e-03
   4.80362133e-03   6.88101139e-01]
------------------------

I should be seeing a 5
I think I'm seeing a 5
[  5.02071478e-03   7.72841968e-03   1.72115426e-03   4.61560899e-03
   7.95180504e-04   9.46171803e-01   1.66947892e-03   1.03414063e-02
   3.03063274e-02   9.30268139e-03]
------------------------

I should be seeing a 6
I think I'm seeing a 6
[  3.62470306e-02   2.20208613e-03   1.03230892e-03   4.76951603e-03
   6.92084738e-03   5.79356388e-03   9.89072965e-01   7.77001200e-04
   3.41236801e-03   6.38016004e-04]
------------------------
```
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

## New Approach
Rotate the training images by 15° and -15° to triple the size of the train data.<br>

Sample Number | Learning Rate | Success Rate | Failure Rate
:--------------:|---------------|--------------|--------------
1 | 0.125 | 94.940 % | 5.060 %
2 | 0.15 | 94.940 % | 5.060 %
3 | 0.05 | 95.410 % | 4.590 %
4 | 0.03 | 94.960 % | 5.040 %
5 | 0.04 | 94.860 % | 5.140 % 
6 | 0.055 | 95.060 % | 4.940 %
