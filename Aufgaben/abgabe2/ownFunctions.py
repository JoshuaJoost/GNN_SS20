__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.5"
__status__ = "Testing" # TODO test

# kernel imports
import numpy as np
import random
import math

# own data imports
import constants
from constants import xMin, xMax, inputNeurons, invalidTrainDataMaxPoint, invalidTrainDataMinPoint, invalidTrainDataExklusivPointDistance, validDataValue, invalidDataValue


# TODO generate Train Data
def edgeOfUnitCircle(points=1):
    edgePoints = np.zeros((points, 2))
    
    alpha = np.arange(0, 360, 360/points)
    for i in range(edgePoints.shape[0]):
        edgePoints[i][0] = math.sin(alpha[i])
        edgePoints[i][1] = math.cos(alpha[i])
        pass

    return edgePoints
    pass

def withinUnitCircle(points=1):

    pass

def outsideUnitCircle(points=1):

    pass

def validData(trainData=1):
    #achtegeben, dass nur anzahl trainData zurückgegeben wird
    #edgeOfUnitCircle(trainData)
    #withinUnitCircle(trainData)

    #return validData
    pass

def validDataLabeld(trainData=1):
    #return labelData(validData(trainData), validDataValue)
    pass

def invalidData(trainData=1):
    #outsideUnitCircle(trainData)

    #return invalidData
    pass

def invalidDataLabeld(trainData=1):
    #return labelData(invalidData(trainData), invalidDataValue)
    pass

def trainData_shuffeld(trainData=1):
    #validDataLabeld
    #invalidDataLabeld

    #return shuffeldTrainData
    pass

# Only this function is called to generate random weights
def generateRandomWeights(startValue, endValue, numberOfWeights):
    return generateRandomWeights_NormalDistributionsCenter(startValue, endValue, numberOfWeights)
    pass

# distribution of the random weights random
def generateRandomWeights_standard(startValue, endValue, numberOfWeights):
    weights = np.zeros(numberOfWeights)

    for i in range(weights.shape[0]):
        weights[i] = np.random.uniform(startValue, endValue)
        pass
    return weights
    pass

# Equal distribution of the random weight around the mean
def generateRandomWeights_NormalDistributionsCenter(startValue, endValue, numberOfWeights):
    weights = np.zeros(numberOfWeights)
    distributionCenter = (startValue + endValue) / 2
    
    firstHalf = 0
    lastHalf = 0

    if numberOfWeights % 2 == 0:
        firstHalf = int(numberOfWeights / 2)
        lastHalf = int(numberOfWeights / 2)
        pass
    else:
        firstHalf = int(numberOfWeights / 2)
        lastHalf = int(numberOfWeights / 2 + 1)
        pass

    for i in range(firstHalf):
        weights[i] = np.random.uniform(distributionCenter, endValue)
        pass

    for i in range(lastHalf):
        weights[firstHalf + i] = np.random.uniform(startValue, distributionCenter)
        pass

    np.random.shuffle(weights)
    return weights
    pass

print(edgeOfUnitCircle(10))

