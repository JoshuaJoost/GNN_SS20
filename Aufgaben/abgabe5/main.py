__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"

# kernel import
import numpy as np
import math
from matplotlib import pyplot as plt

# constants
## functions
tanh_func = lambda netInput: (2 / (1 + math.e ** (-2.0 * float(netInput))) -1)

## net constants
numBias = 1
numNeurons = 2

## weightMatrix
# [[W11, W21],[W12, W22]]
weightMatrix = np.array([[-4, -1.5], [1.5, 0]])
# [WBias1, WBias2]
weightMatrixBias = np.array([-3.37, 0.125])

# [Neuron1, Neuron2]
neuronValues = np.array([0.0,0.0])

# Implements reccurent Hopfield-Net
# main
iterations = 10

plotValuesN1 = np.zeros([iterations + 1])
plotValuesN1[0] = neuronValues[0]

plotValuesN2 = np.zeros([iterations + 1])
plotValuesN2[0] = neuronValues[1]

for i in range(iterations):
    # synchron activation
    synchNeuron1Value = neuronValues[0] * weightMatrix[0][0] + neuronValues[1] * weightMatrix[1][0] + weightMatrixBias[0]
    synchNeuron2Value = neuronValues[0] * weightMatrix[0][1] + neuronValues[1] * weightMatrix[1][1] + weightMatrixBias[1]
    
    neuronValues[0] = tanh_func(synchNeuron1Value)
    neuronValues[1] = tanh_func(synchNeuron2Value)

    plotValuesN1[i+1] = neuronValues[0]
    plotValuesN2[i+1] = neuronValues[1]   
    pass

plt.plot(plotValuesN1, 's-', markersize=6, color='red')
plt.plot(plotValuesN2, 's-', markersize=6, color='blue')
plt.show()


