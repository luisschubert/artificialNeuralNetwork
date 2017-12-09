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
            m = self.initializeMatrix(m, numberOfNodesInLayers[i])
            self.weightMatrices.append(m)
        pass

    def train(self):
        mnist = file("/Users/Student/Desktop/mnist_train.csv")
        for line in iter(mnist):
            pixels = line.split(",")
            digit = pixels.pop(0)
            inputVector = numpy.array(map(float,pixels))
            # scale the input vector to be values between (0,1]
            inputVector = inputVector / 255 * 0.99 +0.01
            # TODO: what format to use here???
            resultantVector = self.forwardPropagate(inputVector)
            self.backwardPropagate(digit, resultantVector)
        mnist.close()

        pass


    def query(self):
        pass


    '''
    HELPER FUNCTIONS 
    '''
    def forwardPropagate(self, inputVector):
        w1 = self.weightMatrices[0].dot(inputVector)
        w2 = self.weightMatrices[1].dot(w1)
        return w2

    def backwardPropagate(self, digit, resultantVector):
        pass

    def initializeMatrix(self, m, numOfNodes):
        # w =
        # i = 0
        for row in range(len(m)):
            m[row] = numpy.random.randn(numOfNodes) * sqrt(2.0/numOfNodes)
            # for col in range(len(m[0])):
            #     m[row][col] = w[i]
            #     i = i + 1
        return m

ann = artificialNeuralNetwork(numberOfLayers=3, numberOfNodesInLayers=(784,100,10), numberOfTrainingIterations=100, learningRate=1)
ann.train()

pass
