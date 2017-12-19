from ANN import artificialNeuralNetwork
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def parseCifarBatch():
    pass



folderPrefix = '/Users/Student/Desktop/cifar-10-batches-py/'
metaInformation = unpickle(folderPrefix+'batches.meta')
batch1 = unpickle(folderPrefix+'data_batch_1')
batch2 = unpickle(folderPrefix+'data_batch_2')
batch3 = unpickle(folderPrefix+'data_batch_3')
batch4 = unpickle(folderPrefix+'data_batch_4')
batch5 = unpickle(folderPrefix+'data_batch_5')


ann = artificialNeuralNetwork(numberOfLayers=3, numberOfNodesInLayers=(3072,100,10), learningRate=0.32)
#TODO: overwrite the train and the query functions.
ann.train(10)
ann.query()
pass
