__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development" # TODO status changes only when other files are implemented

# kernel imports
import scipy.special 

## Value specifications Task 2 ---------------------------
# Don't change this values permanently
inputNeurons = 2 # (x,y) Point
biasNeurons = 1 # 1 per Neuron
hiddenNeurons = 4 # 1 Hidden Layer
outputNeurons = 1 # x and y between ]1,-1[ output 0.8 otherwise 0.0

activationFunc = lambda x: scipy.special.expit(x) # expit = sigmoidfunction

xMax = 1
xMin = -1
xRange = abs(xMax) + abs(xMin)
yMax = 1
yMin = -1
yRange = abs(yMax) + abs(yMin)

invalidDataValue = 0.0
validDataValue = 0.8

## Static values -----------------------------------------
learningRate = 0.01

numberOfValidTrainData = 1000
numberOfInvalidTrainData = 1000

rangeInvalidTrainData = 2
invalidTrainDataMinPoint = xMin - rangeInvalidTrainData
invalidTrainDataMaxPoint = xMax + rangeInvalidTrainData
invalidTrainDataExklusivPointDistance = 0.001

## Labels ------------------------------------------------
INPUT_LAYER = "---- Input Layer ----"
HIDDEN_LAYER = "---- Hidden Layer ----"
OUTPUT_LAYER = "---- Output Layer ----"








