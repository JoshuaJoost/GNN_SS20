__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-05-01"
__version__ = "0.0"
__status__ = "Development" 

# kernel imports
import numpy as np

# own data imports
from neuron import neuron

class neuronalNetworkLayer():

    def __init__(self, numberOfBiasNeurons, numberOfNeurons, layerName, connectionToNextLayer=None, inputLayerInputs=None, isInputLayer=False):
        self.numberOfBiasNeurons = numberOfBiasNeurons
        self.numberOfNeurons = numberOfNeurons
        self.layerName = layerName
        self.connectionToNextLayer = connectionToNextLayer
        self.inputLayerInputs = inputLayerInputs
        self.isInputLayer = isInputLayer

        # build layer
        self.layerNeurons = self.__buildLayerNeurons()

        pass

    def __buildLayerNeurons(self):
        layer = np.empty(self.numberOfBiasNeurons + self.numberOfNeurons, dtype=object)

        for i in range(self.numberOfBiasNeurons):
            layer[i] = neuron(layerName=self.layerName, layerNeuronNumber=i, isInputNeuron=self.isInputLayer, isBiasNeuron=True)
            pass 

        for i in range(self.numberOfNeurons):
            if self.isInputLayer:
                layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons, isInputNeuron=self.isInputLayer, isBiasNeuron=False, input=self.inputLayerInputs[i])
                pass
            else:
                layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons, isInputNeuron=self.isInputLayer, isBiasNeuron=False)
                pass
            pass

        return layer
        pass

    # TODO
    def __buildLayerConnectionsToNextLayer(self):

        pass

    def getLayerNeurons(self):
        return self.layerNeurons
        pass

    def __str__(self):
        layerString = self.layerName + "\n"

        for i in range(self.layerNeurons.shape[0]):
            layerString += self.layerNeurons[i].__str__()

            if i + 1 < self.layerNeurons.shape[0]:
                layerString += "\n"
                pass
            pass

        return layerString
        pass

    pass

inputLayerInputs = np.array([2,3])
inputLayer = neuronalNetworkLayer(1, 2, "InputLayer", isInputLayer=True, inputLayerInputs=inputLayerInputs)
print(inputLayer.__str__())