__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"
##--- TODO
#- Datei ist fertig, wenn Projekt abgabebereit

# kernel imports
import scipy.special 
import numpy as np
import math

## Value specifications Task 2 ---------------------------
# Don't change this values permanently
inputNeurons = 2 # (x,y) Point
biasNeurons = 1 # 1 per layer
hiddenNeurons = 4 # 1 Hidden Layer
outputNeurons = 1 # x and y between ]1,-1[ output 0.8 otherwise 0.0

# Activation functions
sigmoid = lambda x: round(1.0 / (1.0 + round(math.e**-round(float(x), 6), 6)), 6)

# Activation functions derived (for backpropagation step)
sigmoidDerived_1 = lambda x: sigmoid(float(x)) * (1.0 - sigmoid(float(x)))

# Activation function choosed
activationFunction = lambda x: sigmoid(float(x))
activationFunctionDerived_1 = lambda x: sigmoidDerived_1(float(x))

xMax = 3
xMin = -3
xRange = abs(xMax) + abs(xMin)
yMax = 3
yMin = -3
yRange = abs(yMax) + abs(yMin)

invalidDataValue = 0.0
validDataValue = 0.8

## Static values -----------------------------------------
##-- Neuronal Network Values
learningRate = 0.05

weightsMinValue = -1.0
weightsMaxValue = 1.0

##-- Errorfunctions
meanSquaredError = lambda targetValue, outputValue: ((targetValue - outputValue)**2).mean(axis=0)
differenzError = lambda targetValue, outputValue: targetValue - outputValue

# NeuralNetwork accesses this
errorfunction = differenzError

##-- Traindata Values
numberOfValidTrainData = 1000
numberOfInvalidTrainData = numberOfValidTrainData

radiusIntervalCloseToUnicircleBorder = np.array([1.001, 1.2])


# TODO noch relevant?
rangeInvalidTrainData = 2
invalidTrainDataMinPoint = xMin - rangeInvalidTrainData
invalidTrainDataMaxPoint = xMax + rangeInvalidTrainData
invalidTrainDataExklusivPointDistance = 0.001

## Labels ------------------------------------------------
inputLayerLabel = "---- Input Layer ----"
hiddenLayerLabel = "---- Hidden Layer ----"
outputLayerLabel = "---- Output Layer ----"








