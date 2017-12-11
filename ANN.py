# coding=utf-8
import numpy
from math import sqrt
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
            m = self.initializeMatrix(m, numberOfNodesInLayers[i])
            self.weightMatrices.append(m)
        pass

    def train(self):
        mnist = file("/Users/luisschubert/Downloads/mnist_train.csv")
        imageCount = 0
        for line in iter(mnist):
            if imageCount == 100:
                pass
                # break
            pixels = line.split(",")
            digit = int(pixels.pop(0))
            inputVector = numpy.array(map(float,pixels))
            # scale the input vector to be values between (0,1]
            inputVector = inputVector / 255 * 0.99 + 0.01
            # TODO: what format to use here???
            layers = self.forwardPropagate(inputVector)
            self.backwardPropagate(digit, layers)
            imageCount = imageCount + 1
            print('Number of digits seen: %s' % str(imageCount))
            # print("I am learning to see a %s" % str(digit))
            # print(layers['sigOutputNodes'])
            # print("------------------------\n")
        mnist.close()

        pass


    def query(self):
        mnistTest = file("/Users/luisschubert/Downloads/mnist_test.csv")
        imageCount = 0
        correct = 0
        incorrect = 0

        for line in iter(mnistTest):
            if imageCount == 100:
                pass
            pixels = line.split(",")

            '''
            Prepare Input
            '''
            digit = int(pixels.pop(0))
            inputVector = numpy.array(map(float, pixels))
            # scale the input vector to be values between (0,1]
            inputVector = inputVector / 255 * 0.99 + 0.01

            '''
            Feed the Network
            '''
            layers = self.forwardPropagate(inputVector)

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
            Reporting
            '''
            print("I should be seeing a %s"% str(digit))
            print("I think I'm seeing a %s" % str(recognizedDigit))
            print(layers['sigOutputNodes'])
            print("------------------------\n")


            '''
            Bookkeeping
            '''
            imageCount = imageCount + 1

        '''
        Diagnostics:
        '''
        print('successRate: %.3f' % ((float(correct)/float(imageCount))*100) )
        print('failureRate: %.3f' % ((float(incorrect)/float(imageCount))*100) )


        '''
        Housekeeping
        '''
        mnistTest.close()
        pass


    '''
    HELPER FUNCTIONS 
    '''
    def forwardPropagate(self, inputVector):
        layers = {}
        layers['inputNodes'] = inputVector
        # inputVector can also be thought of as the inputLayer.
        # the inputLayer is composed of 784 nodes 1 for each of the 784 pixels.

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

    # TODO:
    # TEST THIS ASDSDADASFASDASDAS
    def backwardPropagate(self, digit, layers):
        expectedVector = numpy.zeros((10))
        # This vector: [0,0,0,0,0,1,0,0,0,0] might not be in the right format
        # might need to be [0.01,0.01,0.01,0.01,0.01,0.99,0.01,0.01]
        expectedVector[digit] = 1
        expectedVector = expectedVector * 0.99 + 0.01

        '''
        really stuck here.
        can't figure out the proper implementation for the backwardsPropagation.
        steps:
        1. calculate the error for all the outputLayer values.
        errorOutputLayer = resultantVector - expectedVector 
        2. calculate the delta for the outputWeightMatrix (number of layers - 1; self.weightMatrices[1])
        hiddenLayer(1x100)
        outputLayer(1x10)
        outputLayerError(1x10)
        learningRate(scalar)
        outputWeightMatrix(10x100)
        newOutputWeightMatrix(10x100)
        s =  outputLayer*(1 - outputLayer) (1x10)
        errS = errorOutputLayer * s (1x10)
        alphaErrS = learningRate * errS (1x10)
        
        newOutputWeightMatrix = outputWeightMatrix - learningRate * (outputLayerError * outputLayer*(1 - outputLayer)) • hiddenLayer
        newOutputWeightMatrix = outputWeightMatrix - learningRate * (outputLayerError * s) • hiddenLayer
        newOutputWeightMatrix = outputWeightMatrix - learningRate * errS • hiddenLayer
        newOutputWeightMatrix = outputWeightMatrix - alphaErrS • hiddenLayer
        
        3. calculate the error for all the hiddenLayer values.
        4. calculate the delta for the hiddenWeightMatrix (number of layers - 2; self.weightMatrices[0])
        '''
        # calculate error between hidden and output layer
        # slopeOfErrorFunctionJK = -errorOutputLayer * numpy.dot( self.sigmoidDerivOfValue(layers['sigOutputNodes']),layers['hiddenNodes'])
        # deltaWjk = self.learningRate * numpy.inner(numpy.transpose(errorOutputLayer), (numpy.sum(self.sigmoidDerivOfValue(layers['sigOutputNodes'])) * layers['sigHiddenNodes']))
        # newWjk = self.weightMatrices[1] - self.learningRate * slopeOfErrorFunctionJK
        pass

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


        # newWij = self.weightMatrices[0] - self.learningRate * (numpy.dot((numpy.dot(-errorHiddenLayer, self.sigmoidDerivOfValue(layers['sigHiddenNodes']))), layers['hiddenNodes']))

        # errorsInputLayer = numpy.transpose(self.weightMatrices[0]) * errorsHiddenLayer

        # calculate error between input and hidden layer

        pass

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def sigmoidDerivOfValue(self, x):
        return x * (1 - x)

    def calculateError(self, resultantVector, expectedVector):
        return (expectedVector - resultantVector)

    def initializeMatrix(self, m, numOfNodes):
        # w =
        # i = 0
        for row in range(len(m)):
            m[row] = numpy.random.randn(numOfNodes) * sqrt(2.0/numOfNodes)
            # for col in range(len(m[0])):
            #     m[row][col] = w[i]
            #     i = i + 1
        return m



ann = artificialNeuralNetwork(numberOfLayers=3, numberOfNodesInLayers=(784,100,10), learningRate=0.5)
ann.train()
ann.query()

pass
