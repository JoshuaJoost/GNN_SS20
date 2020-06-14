#Run cell 
#%%

__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-21"
__version__ = "0.5"
__status__ = "Test"

import numpy as np 
import matplotlib
from matplotlib import pyplot as plt

print(f"numpy_version: {np.version.version}")
print(f"matplotlib version: {matplotlib.__version__}")

## Value specifications Task 1 --------------------------------
# Don't change this values
DELTA_T = 0.01
X_VALUES = np.array([-7.0, -0.2, 8.0])

## static diagram values --------------------------------------
# Points to sample the curve
numberOfSamplePoints = 64

# diagram dimensions, have to be constant 2
DIAG_DIM = 2

# diagram range
xMin = -4
xMax = 4
# y value function
yFunc = lambda x: x - (x ** 3)

## delta train function ------------------------------------------
trainSamplingPoints = numberOfSamplePoints
deltaLernFunc = lambda x, delta_t: x + delta_t * (x - x ** 3)

def generateTrainValues(startValue, trainIterations = trainSamplingPoints, trainFunc = deltaLernFunc):
    trainValues = np.zeros(trainIterations)

    trainValues[0] = trainFunc(startValue, DELTA_T)
    for i in range(1, trainIterations):
        trainValues[i] = trainFunc(trainValues[i-1], DELTA_T)
    
    return trainValues

## Calculate plot values ---------------------------------------- 
# initialise diagram values array
pltValues = np.zeros((DIAG_DIM, numberOfSamplePoints))

# generate xValues
pltValues[0] = np.linspace(xMin, xMax, numberOfSamplePoints, endpoint=True)

# generate yValues
for yi in range(numberOfSamplePoints):
    pltValues[1][yi] = yFunc(pltValues[0][yi])

# print(pltValues)

## Calculate delta train values --------------------------------
trainValues = np.zeros((X_VALUES.size, trainSamplingPoints))
for i in range(X_VALUES.size):
    trainValues[i] = generateTrainValues(X_VALUES[i])

#print(trainValues)

## plot diagram --------------------------------------------------
xMinValue = xMin
xMaxValue = xMax
yMinValue = np.min(pltValues[1])
yMaxValue = np.max(pltValues[1])

plt.plot(pltValues[0], pltValues[1], label = 'basic function')
plt.plot(pltValues[0], trainValues[0], marker='o', markersize=2, color='green', label=X_VALUES[0], alpha=0.8)
plt.plot(pltValues[0], trainValues[1], marker='o', markersize=2, color='red', label=X_VALUES[1], alpha=0.8)
plt.plot(pltValues[0], trainValues[2], marker='o', markersize=2, color='blue', label=X_VALUES[2], alpha=0.8)
plt.axis([xMinValue, xMaxValue, yMinValue + (yMinValue / 10), yMaxValue + (yMaxValue / 10)])
plt.legend()
plt.show()

#  Attraktor läuft auf einen Fixpunkt zu (in Richtung des Wertes des Sattelpunkts y = 0)
