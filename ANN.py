# coding=utf-8


import numpy
from math import sqrt
import time
class NonMatchingLayersAndNodeNumbers(Exception):
    pass

class artificialNeuralNetwork:
    # define numberOfNodesInLayers as a n-tuple of natural numbers where n == numberOfLayers
    def __init__(self, numberOfLayers, numberOfNodesInLayers, learningRate):
        if numberOfLayers != len(numberOfNodesInLayers):
            raise NonMatchingLayersAndNodeNumbers('[numberOfLayers: %d, numberOfNodesInLayers: %d]' % (numberOfLayers, len(numberOfNodesInLayers)))
        self.learningRate = learningRate
        self.weightMatrices = []
        for i in range(numberOfLayers-1):
            m = numpy.zeros((numberOfNodesInLayers[i+1], numberOfNodesInLayers[i]))
            m = self.initializeMatrixForReLUNeurons(m, numberOfNodesInLayers[i])
            self.weightMatrices.append(m)
        pass

    def MNISTtrain(self, epoch=1):
        # https://www.pjreddie.com/media/files/mnist_train.csv
        timeElapsed = 0
        for epochNumber in range(epoch):
            start  = time.time()
            print ("epoch number: %d" % (epochNumber + 1))
            with open("/Users/Student/Desktop/mnist_train.csv", "r") as mnist:
                imageCount = 0
                for line in mnist:
                    pixels = line.split(",")
                    digit = int(pixels.pop(0))
                    #inputVector = numpy.array(map(float,pixels))
                    # Use this for python 3.7
                    inputVector = numpy.array(pixels).astype(numpy.float)
                    # scale the input vector to be values between (0,1]
                    inputVector = inputVector / 255 * 0.99 + 0.01
                    layers = self.threeLayerForwardPropagate(inputVector)
                    self.threeLayerBackwardPropagate(digit, layers)
                    imageCount = imageCount + 1
                    if imageCount % 1000 == 0:
                        print('Number of digits seen: %s' % str(imageCount))
                mnist.close()
                end = time.time()
                iterationTime = end - start
                timeElapsed = timeElapsed + iterationTime
                print ("iteration Time %s" % str(iterationTime))
                print ("elapsed time %s" % str(timeElapsed))
        pass



    def MNISTquery(self):
        # https://www.pjreddie.com/media/files/mnist_test.csv
        with open("/Users/Student/Desktop/mnist_test.csv", "r") as mnistTest:
            imageCount = 0
            correct = 0
            incorrect = 0

            for line in mnistTest:
                '''
                Prepare Input
                '''
                pixels = line.split(",")
                digit = int(pixels.pop(0))
                #inputVector = numpy.array(map(float, pixels))
                inputVector = numpy.array(pixels).astype(numpy.float)
                # scale the input vector to be values between (0,1]
                inputVector = inputVector / 255 * 0.99 + 0.01


                '''
                Feed the Network
                '''
                layers = self.threeLayerForwardPropagate(inputVector)


                '''
                Interpret Output
                '''
                recognizedDigit = numpy.argmax(layers['sigOutputNodes'])


                '''
                Error Calculation
                '''
                if recognizedDigit == digit:
                    correct = correct + 1
                else:
                    incorrect = incorrect + 1
                    '''
                    Error Reporting
                    '''
                    print("I should be seeing a %s"% str(digit))
                    print("I think I'm seeing a %s" % str(recognizedDigit))
                    print(layers['sigOutputNodes'])
                    print("------------------------\n")
                imageCount = imageCount + 1

            '''
            Diagnostics:
            '''
            print('successRate: %.3f' % ((float(correct)/float(imageCount))*100) )
            print('failureRate: %.3f' % ((float(incorrect)/float(imageCount))*100) )
            mnistTest.close()
            pass


    '''
    HELPER FUNCTIONS 
    '''

    # This is currently hardcoded for just 1 hidden layer.
    # Todo: Fix this and implement the general case of N-Hidden Layers
    def threeLayerForwardPropagate(self, inputVector):
        layers = {}
        layers['inputNodes'] = inputVector

        # w1 is the layer representing the hiddenLayer.
        # the hiddenLayer is composed of 100 nodes.
        layers['hiddenNodes'] = self.weightMatrices[0].dot(inputVector)
        # the sigmoid function is applied before passing the signals to the next layer.
        layers['sigHiddenNodes'] = self.sigmoid(layers['hiddenNodes'])

        # w2 is the layer representing the outputlayer.
        # the outputLayer is composed of 10 nodes.
        layers['outputNodes'] = self.weightMatrices[1].dot(layers['sigHiddenNodes'])
        # the sigmoid function is applied before interpreting the signals as the output of the network.
        layers['sigOutputNodes'] = self.sigmoid(layers['outputNodes'])


        # return the layers
        return layers


    # This is currently hardcoded for just 1 hidden layer.
    # Todo: Fix this and implement the general case of N-Hidden Layers
    def threeLayerBackwardPropagate(self, digit, layers):
        expectedVector = numpy.zeros((10))
        expectedVector[digit] = 1
        expectedVector = expectedVector * 0.99 + 0.01
        
        
        '''
        Update Weights for (hidden)-OutputMatrix
        '''
        errorOutputLayer = self.calculateError(layers['sigOutputNodes'], expectedVector)
        deltaOutputWeightMatrix = numpy.outer(
            self.learningRate * (
                -errorOutputLayer * (
                    layers['sigOutputNodes'] * (1 - layers['sigOutputNodes'])
                )
            ),
                    layers['sigHiddenNodes']
        )
        self.weightMatrices[1] = self.weightMatrices[1] - deltaOutputWeightMatrix

        
        '''
            Update Weights for (input)-HiddenMatrix
        '''
        errorHiddenLayer = numpy.dot(numpy.transpose(self.weightMatrices[1]), errorOutputLayer)
        deltaHiddenWeightMatrix = numpy.outer(
            self.learningRate * (
                -errorHiddenLayer * (
                    layers['sigHiddenNodes'] * (1 - layers['sigHiddenNodes'])
                )
            ),
                    layers['inputNodes']
        )
        self.weightMatrices[0] = self.weightMatrices[0] - deltaHiddenWeightMatrix
        pass

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def sigmoidDerivOfValue(self, x):
        return x * (1 - x)

    def calculateError(self, resultantVector, expectedVector):
        return (expectedVector - resultantVector)

    def initializeMatrixForReLUNeurons(self, m, numOfNodes):
        for row in range(len(m)):
            m[row] = numpy.random.randn(numOfNodes) * sqrt(2.0/numOfNodes)
        return m
