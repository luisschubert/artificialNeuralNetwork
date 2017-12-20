from ANN import artificialNeuralNetwork

ann = artificialNeuralNetwork(numberOfLayers=3, numberOfNodesInLayers=(784,100,10), learningRate=0.3)
ann.MNISTtrain(5)
ann.MNISTquery()
