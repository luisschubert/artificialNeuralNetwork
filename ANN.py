import numpy
from math import sqrt
class NonMatchingLayersAndNodeNumbers(Exception):
    pass

class artificialNeuralNetwork:
    # define numberOfNodesInLayers as a n-tuple of natural numbers where n == numberOfLayers
    def __init__(self, numberOfLayers, numberOfNodesInLayers, numberOfTrainingIterations, learningRate):
        if numberOfLayers != len(numberOfNodesInLayers):
            raise NonMatchingLayersAndNodeNumbers('[numberOfLayers: %d, numberOfNodesInLayers: %d]' % (numberOfLayers, len(numberOfNodesInLayers)))
        self.numberOfTrainingIterations = numberOfTrainingIterations
        self.learningRate = learningRate
        self.weightMatrices = []
        for i in range(numberOfLayers-1):
            m = numpy.zeros((numberOfNodesInLayers[i+1], numberOfNodesInLayers[i]))
            self.
            self.weightMatrices.append(m)
        pass

    def train(self):
        mnist = file("MNIST/mnist_train.csv")
        for line in iter(mnist):
            pixels = line.split(",")
            digit = pixels.pop(0)
            # TODO: what format to use here???
            resultantVector = self.forwardPropagate(pixels)
            self.backwardPropagate(digit, resultantVector)
        mnist.close()

        pass


    def query(self):
        pass


    '''
    HELPER FUNCTIONS 
    '''
    def forwardPropagate(self, inputVector):
        pass

    def backwardPropagate(self, digit, resultantVector):
        pass

    def initializeMatrix(self, m, numOfNodes):
        for row in range(len(m)):
            for col in range(len(m[0])):
                m[row][col] = numpy.random.randn(numOfNodes) * sqrt(2.0/numOfNodes)

ann = artificialNeuralNetwork(3,(784,100,10), 100)
