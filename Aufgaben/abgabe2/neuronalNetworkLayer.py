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
from ownFunctions import generateRandomWeights
from constants import weightsMinValue, weightsMaxValue

class neuronalNetworkLayer():

    def __init__(self, numberOfBiasNeurons, numberOfNeurons, layerName, inputLayerInputs=None, isInputLayer=False, isOutputLayer=False):
        self.numberOfBiasNeurons = numberOfBiasNeurons
        self.numberOfNeurons = numberOfNeurons
        self.layerName = layerName
        self.inputLayerInputs = inputLayerInputs
        self.isInputLayer = isInputLayer
        self.isOutputLayer = isOutputLayer

        # Check for faulty layer typing
        if self.isInputLayer and self.isOutputLayer:
            raise ValueError("Layer kann nicht gleichzeitig Input- und Outputlayer sein")

        # ignore biases at output layer
        if self.isOutputLayer and self.numberOfBiasNeurons != 0:
            self.numberOfBiasNeurons = 0

        # build layer
        self.layerNeurons = self.__buildLayerNeurons()

        # --- Connection to next Layer
        self.weights = None
        self.connectToLayer = None

        pass

    def __buildLayerNeurons(self):
        layer = np.empty(self.numberOfBiasNeurons + self.numberOfNeurons, dtype=object)

        for i in range(self.numberOfBiasNeurons):
                layer[i] = neuron(layerName=self.layerName, layerNeuronNumber=i+1, isBiasNeuron=True)
                pass 

        for i in range(self.numberOfNeurons):
            if self.isInputLayer:
                layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons+1, isInputNeuron=self.isInputLayer, input=self.inputLayerInputs[i])
                pass
            elif self.isOutputLayer:
                # Outputlayer has no bias Neurons
                layer[i] = neuron(layerName=self.layerName, layerNeuronNumber=i+1, isOutputNeuron=True)
                pass
            else:
                layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons+1)
                pass
            pass

        return layer
        pass

    def connectTo(self, nextLayer):
        self.weights = np.zeros((self.numberOfBiasNeurons + self.numberOfNeurons, nextLayer.numberOfNeurons))
        self.connectToLayer = nextLayer

        pass

    def setRandomWeights(self, weightsMin=weightsMinValue, weightsMax=weightsMaxValue):
        randomWeights = generateRandomWeights(weightsMin, weightsMax, self.weights.size)
        
        for row in range(self.weights.shape[0]):
            for column in range(self.weights.shape[1]):
                self.weights[row][column] = randomWeights[row * self.weights.shape[1] + column]
                pass
            pass

        pass

    def calcInputsNextLayer():

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

    def setInputLayerInputs(self, newInputs):
        if self.isInputLayer:
            for i in range(self.numberOfNeurons):
                self.layerNeurons[self.numberOfBiasNeurons + i] = newInputs[i]
                pass
        pass

    pass

inputLayerInputs = np.array([2,3])
inputLayer = neuronalNetworkLayer(1, 2, "InputLayer", isInputLayer=True, inputLayerInputs=inputLayerInputs)
hiddenLayer = neuronalNetworkLayer(1, 4, "HiddenLayer")
outputLayer = neuronalNetworkLayer(0, 1, "OutputLayer", isOutputLayer=True)
print(inputLayer.__str__())
print(hiddenLayer.__str__())
print(outputLayer.__str__())

inputLayerInputs = np.array([7,90])
inputLayer.setInputLayerInputs(inputLayerInputs)
print(inputLayer.__str__())
#inputLayer.connectTo(hiddenLayer)
#inputLayer.setRandomWeights()
















