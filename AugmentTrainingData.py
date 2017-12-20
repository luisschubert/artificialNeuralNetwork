'''
Needs Python 3.5 for PIL I belive.

'''
from PIL import Image

import numpy

def numpyToCSString(arr):
    s = ''
    for i in range(arr.size):
        s = s +(str(arr[i]))
        if i != arr.size - 1:
            s = s + ","
        else:
            s = s + "\n"
    return s

with open('/PATH/TO/mnist_train.csv', "r") as inFile,\
        open('/PATH/TO/mnist_trainRotated.csv', "w") as outFile:
    imageRead = 1
    for line in inFile:
        imagePixelData = line.split(',')
        digit = int(imagePixelData.pop(0))
        pixelValues = numpy.array(imagePixelData).astype(numpy.uint8)
        blank = numpy.zeros((28,28)).astype(numpy.uint8)
        count = 0
        for i in range(28):
            for j in range(28):
                blank[i][j] = pixelValues[count]
                count = count + 1
        im = Image.fromarray(blank, mode="L")

        toRightImage = im.rotate(15)
        toRightValues = numpy.array(toRightImage).flatten()
        toRightValues = numpy.insert(toRightValues, 0, digit)

        toLeftImage = im.rotate(-15)
        toLeftValues = numpy.array(toLeftImage).flatten()
        toLeftValues = numpy.insert(toLeftValues, 0, digit)


        outFile.write(numpyToCSString(toRightValues))
        outFile.write(numpyToCSString(toLeftValues))
        print(imageRead)
        imageRead = imageRead + 1
