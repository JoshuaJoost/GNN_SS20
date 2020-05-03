__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-05-01"
__version__ = "0.0"
__status__ = "Development" 
##--- TODO
# - testen
# - [optional]: setzen spezifischer Gewichte 

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
                if isinstance(self.inputLayerInputs, type(None)):
                    layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons+1, isInputNeuron=self.isInputLayer)
                    pass
                else:
                    layer[i + self.numberOfBiasNeurons] = neuron(layerName=self.layerName, layerNeuronNumber=i+self.numberOfBiasNeurons+1, isInputNeuron=self.isInputLayer, input=self.inputLayerInputs[i])
                    pass
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

    ##--- Weights
    # Switches between the functions for setting the weights
    def setWeights(self, useSpecificWeights=False, specificWeightsArray=None, generateRandomWeights=False, randomWeightsMinValue=weightsMinValue, randomWeightsMaxValue=weightsMaxValue):
        if useSpecificWeights == generateRandomWeights == False:
            raise ValueError("'useSpecificWeights' oder 'generateRandomWeights' muss auf True gesetz sein")
            pass
        elif useSpecificWeights == generateRandomWeights == True:
            raise ValueError("Nur einer der Parameter 'useSpecificWeights','generateRandomWeights' darf auf True gesetzt sein")
            pass
        elif useSpecificWeights == True and isinstance(specificWeightsArray, type(None)):
            raise ValueError("Das Nutzen spezifischer Gewichte erfordert das Übergeben spezifischer Gewichte")
            pass

        if useSpecificWeights:
            #TODO
            raise Exception("Derzeit nicht implementiert")
            pass
        elif generateRandomWeights:
            self.setRandomWeights(randomWeightsMinValue, randomWeightsMaxValue)
            pass
        else:
            raise Exception("Methodik zum Setzen der Gewichte fehlgeschlagen")
            pass

        pass

    def setRandomWeights(self, weightsMin, weightsMax):
        randomWeights = generateRandomWeights(weightsMin, weightsMax, self.weights.size)
        
        for row in range(self.weights.shape[0]):
            for column in range(self.weights.shape[1]):
                self.weights[row][column] = randomWeights[row * self.weights.shape[1] + column]
                pass
            pass

        pass

    def getLayerWeights(self):
        return self.weights
        pass

    def calcInputsNextLayer(self):
        inputsNextLayer = np.dot(self.getLayerNeuronsInputValues().T, self.weights)
        
        return inputsNextLayer
        pass

    def setInputsNextLayer(self):
        self.connectToLayer.setLayerInputs(self.calcInputsNextLayer())

        pass

    def getLayerNeurons(self):
        return self.layerNeurons
        pass

    def getLayerNeuronsInputValues(self):
        layerNeuronsInputValues = np.zeros(self.layerNeurons.shape[0])
        
        for i in range(layerNeuronsInputValues.shape[0]):
            layerNeuronsInputValues[i] = self.layerNeurons[i].getInput()
            pass

        return layerNeuronsInputValues
        pass

    def getLayerNeuronsOutputValues(self):
        layerNeuronsOutputValues = np.zeros(self.layerNeurons.shape[0])

        for i in range(layerNeuronsOutputValues.shape[0]):
            layerNeuronsOutputValues[i] = self.layerNeurons[i].getOutput()
            pass

        return layerNeuronsOutputValues
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

    def setLayerInputs(self, newInputs):
        for i in range(self.numberOfNeurons):
            self.layerNeurons[self.numberOfBiasNeurons + i].setInput(newInputs[i])
            pass

    pass

#inputLayerInputs = np.array([4,2])
#inputLayer = neuronalNetworkLayer(1, 2, "InputLayer", isInputLayer=True, inputLayerInputs=inputLayerInputs)
#h1 = neuronalNetworkLayer(1, 4, "HiddenLayer1")
#h2 = neuronalNetworkLayer(1, 4, "HiddenLayer2")
#outputLayer = neuronalNetworkLayer(0, 1, "OutputLayer", isOutputLayer=True)

##-- connections
# connect input and h1
#inputLayer.connectTo(h1)
#inputLayer.setRandomWeights()

# connect h1 and h2
#h1.connectTo(h2)
#h1.setRandomWeights()

# connect h2 and output
#h2.connectTo(outputLayer)
#h2.setRandomWeights()

##-- manual forwarding
# forward input -> h1
#inputLayer.setInputsNextLayer()
#h1.getLayerNeuronsOutputValues()

# forward h1 -> h2
#h1.setInputsNextLayer()
#h2.getLayerNeuronsOutputValues()

# forward h2 -> output
#h2.setInputsNextLayer()
#print(outputLayer.getLayerNeuronsOutputValues())















