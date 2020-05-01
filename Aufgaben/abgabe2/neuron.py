__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-05-01"
__version__ = "0.0"
__status__ = "Development" 

# kernel imports
import numpy as np

# own data imports
from constants import activationSigmoid

class neuron:

    def __init__(self, layerName, layerNeuronNumber, input = 0, isBiasNeuron = False, isInputNeuron = False, activationFunc = activationSigmoid):
        # init neuron via params
        self.isBiasNeuron = isBiasNeuron
        self.isInputNeuron = isInputNeuron
        self.input = input
        self.activationFunc = activationFunc
        self.layerName = layerName
        self.layerNeuronNumber = layerNeuronNumber

        # further init 
        self.neuronName = ""

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
        if self.isInputNeuron:
            if self.isBiasNeuron:
                # bias neurons just returns 1
                return 1
                pass
            else:
                return self.input
                pass
            pass
        else:
            if self.isBiasNeuron:
                return 1
                pass
            else:
                return self.activationFunc(self.input)
                pass
            pass

    def __str__(self):
        return self.neuronName + ": " + str(self.getOutput())
        pass

    pass



inputLayerX = 2
inputLayerY = 3
n1 = neuron(layerName="InputLayer", layerNeuronNumber=1, isInputNeuron=True, isBiasNeuron=True)
print(n1.__str__())















