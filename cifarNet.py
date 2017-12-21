from ANN import artificialNeuralNetwork

ann = artificialNeuralNetwork(numberOfLayers=3, numberOfNodesInLayers=(3072,100,10), learningRate=0.32)
#TODO: overwrite the train and the query functions.
ann.CIFAR10train()
ann.CIFAR10query()
pass
