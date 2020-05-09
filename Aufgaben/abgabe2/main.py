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
from view import printCircle
import neuronalNetwork as nn


## Generate Traindata -------------------------------------
#validTrainData = generateNValidTrainDataLabeld(numberOfValidTrainData)
#invalidTrainData = generateNInvalidTrainDataLabeld(numberOfInvalidTrainData)

## init neuronalnetwork
#nn = neuronalNetwork.neuralNetwork()
#print("def layer")
#inputLayer = np.array([1, 2])
#nHiddenLayer = np.array([[1,4]])
#outputLayer = np.array([1])

#nen = nn.neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)
#trainData = ownFunctions.trainDataLabeld_shuffeld(100000)

#tquery = lambda x,y: nen.forwarding(np.array([x,y])) 
#printCircle(2, tquery)











