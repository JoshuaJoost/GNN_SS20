__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"
##--- TODO
#- generieren der testdaten
#- erstellen des neuronalen Netzes
#- trainieren des neuronalen Netzes
#- Zeichnen der Graphiken (neuronale Netz Einheitskreis zeichnen lassen, etc.)
#- testen

# kernel imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# own data imports
import ownFunctions
import constants
from constants import numberOfValidTrainData, numberOfInvalidTrainData
from view import printCircle, printSummary
import neuronalNetwork as nn




## -- Init neuronalnetwork
inputLayer = np.array([1, 2])
nHiddenLayer = np.array([[1,4]])
outputLayer = np.array([1])

nen = nn.neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)

## -- Training phase
# Generate Traindata
trainData = ownFunctions.getRandomTrainData(150000)

# Train neuronal network
print("Trainingsphase 1")
nen.trainWithlabeldData(trainData)
#print("Trainingsphase 2")
#nen.trainWithlabeldData(trainDataValid)
#print("Trainingsphase 3")
#nen.trainWithlabeldData(trainData)
#print("Trainingsphase 4")
#nen.trainWithlabeldData(trainData)
#print("Trainingsphase 5")
#nen.trainWithlabeldData(trainData)
print("Training beendet")

## -- Test phase
# Generate Testdata
testData = ownFunctions.trainDataLabeld_shuffeld(1)

# Evaluate neuronal network
okValue = 0
print("Output --- Target")
for i in range(testData.shape[0]):
    nnForwardingValue = nen.forwarding(testData[i])
    if abs(testData[i][2] - nnForwardingValue) >= 0 and abs(testData[i][2] - nnForwardingValue) <= 0.2:
        print("\x1B[32m" + str(nnForwardingValue) + " --- " + str(testData[i][2]) + "\x1B[0m")
        okValue += 1
        pass
    else:
        print("\x1B[31m" + str(nnForwardingValue) + " --- " + str(testData[i][2]) + "\x1B[0m")
        pass

    pass
print(str(int(okValue * testData.shape[0] / 100)) + '% richtig')

# prepare plot data error
plotData_Error = nen.preparePlotData_Error(dataDivisor=1000)

# print evaluation


#tquery = lambda x,y: nen.forwarding(np.array([x,y])) 




# dummy data
#dataError = np.array([0.223, 0.212, 0.201, 0.208, 0.210, 0.203, 0.201, 0.199, 0.198, 0.195, 0.196]).dot(100)
#dataPerformance = np.array([0.123, 0.212, 0.302, 0.404, 0.567, 0.654, 0.778, 0.802, 0.823, 0.845, 0.901]).dot(100)

# standalone
#printCircle(2, tquery)

# summary
#printSummary(dataError, dataPerformance, 2, tquery)









