__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-05-01"
__version__ = "0.5"
__status__ = "Test" 
##--- TODO
#- testen

# kernel imports
import numpy as np

# own data imports
from constants import activationFunction

class neuron:

    def __init__(self, layerName, layerNeuronNumber, input = 0, isBiasNeuron = False, isInputNeuron = False, isOutputNeuron=False, activationFunc = activationFunction):
        # init neuron via params
        self.isBiasNeuron = isBiasNeuron
        self.isInputNeuron = isInputNeuron
        self.isOutputNeuron = isOutputNeuron
        self.input = input
        self.activationFunc = activationFunc
        self.layerName = layerName
        self.layerNeuronNumber = layerNeuronNumber

        # further init 
        self.neuronName = ""

        # backpropagation
        self.delta = 0.0

        # if isBias initialise neuron as bias neuron
        if isBiasNeuron:
            self.neuronName = "Bias" + str(self.layerNeuronNumber)
            self.input = 1
            pass
        else:
            self.neuronName = "Neuron" + str(self.layerNeuronNumber)
            pass
        pass

    def getOutput(self):
        if self.isBiasNeuron:
            return 1
            pass
        elif self.isInputNeuron:
            return self.input
            pass
        else:
            return self.activationFunc(self.input)
            pass

        pass


    def __str__(self):
        return self.neuronName + ": " + str(self.getOutput())
        pass

    def setInput(self, newInput):
        self.input = newInput
        pass

    def getInput(self):
        return self.input
        pass

    def setDelta(self, newDeltaValue):
        self.delta = newDeltaValue
        pass

    def getDelta(self):
        return self.delta
        pass

    pass



#inputLayerX = 2
#inputLayerY = 3
#n1 = neuron(layerName="InputLayer", layerNeuronNumber=1, isInputNeuron=False, isBiasNeuron=True, input=inputLayerX)
#print(n1.__str__())















