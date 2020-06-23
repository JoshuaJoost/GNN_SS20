__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"

# kernel import
import numpy as np
import math

# constants
## functions
tanh_func = lambda netInput: (2 / (1 + math.e ** -2.0 * float(netInput)) -1)

## net constants
numBias = 2
numNeurons = 2

## weightMatrix
# [WeightsBias2, WeightsBias1, W11, W12, W21, W22]
weightMatrix = np.array([[-3.37, 0.125, -4, 1.5, -1.5, 0]])

# startingValues [startingValueNeron1, starzingValueNeuron2, Bias1Value, Bias2Value]
startingValues = np.array([[0.0, 0.0, 1, 1]])

# Implements reccurent Hopfield-Net
# main

matrix = np.dot(weightMatrix.T, startingValues)

b1Netinput = np.sum(matrix[0]) + np.sum(matrix[1]) + np.sum(matrix[4])
b2Netinput = np.sum(matrix[0]) + np.sum(matrix[3]) + np.sum(matrix[5])
n1NetInput = np.sum(matrix[2]) + np.sum(matrix[3]) + np.sum(matrix[5])
n2NetInput = np.sum(matrix[2]) + np.sum(matrix[4]) + np.sum(matrix[5])

b1Activate = tanh_func(b1Netinput)
b2Activate = tanh_func(b2Netinput)
n1Activate = tanh_func(n1NetInput)
n2Activate = tanh_func(n2NetInput)

print("Werden alle Neuronen synchron aktiviert erhält man die 4 Werte: [" + str(b1Activate) + "][" + str(b2Activate) + "][" + str(n1Activate) + "][" + str(n2Activate) + "]")





