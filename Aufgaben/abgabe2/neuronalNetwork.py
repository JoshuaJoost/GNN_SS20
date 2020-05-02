__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"

# kernel imports
import numpy as np
import scipy.special 
import types

# own data imports
import constants
from constants import inputNeurons, biasNeurons, hiddenNeurons, outputNeurons, activationSigmoid, learningRate
from constants import inputLayerLabel, hiddenLayerLabel, outputLayerLabel
import ownFunctions
import neuronalNetworkLayer as nnl

### TODO Implement class of the neural network
### The main file calls this

class neuronalNetwork:
    
    # :param2: inputLayerArray: shape(1,numberOfInputNeurons) [0] = BiasNeurons, [1] = InputNeurons
    # :param3: hiddenLayerNDIMArray: shape(numberOfHiddenLayers, 2) [x][0] = NumberOfBiasNeurons, [x][1] = NumberOfNeurons
    # :param4: outputLayerArray: shape(numberOfOutputNeurons) [0] = NumberOfOutputNeurons
    def __init__(self, inputLayerArray, hiddenLayerNDIMArray, outputLayerArray):
        self.neuronalNetworkStructure = np.empty(1 + hiddenLayerNDIMArray.shape[0] + 1, dtype=object)
        #self.neuronalNetworkConnections = None

        # create inputlayer
        self.neuronalNetworkStructure[0] = nnl.neuronalNetworkLayer(inputLayerArray[0], inputLayerArray[1], inputLayerLabel, isInputLayer=True)

        # create hiddenLayer
        for i in range(hiddenLayerNDIMArray.shape[0]):
            self.neuronalNetworkStructure[i + 1] = nnl.neuronalNetworkLayer(hiddenLayerNDIMArray[i][0], hiddenLayerNDIMArray[i][1], hiddenLayerLabel + " (" + str(i+1) + ")")
            pass

        # create outputLayer
        self.neuronalNetworkStructure[-1] = nnl.neuronalNetworkLayer(0, outputLayerArray[0], outputLayerLabel, isOutputLayer=True)

        self.__connectLayers()
        self.__setWeights()

        pass

    def __connectLayers(self):
        for i in range(self.neuronalNetworkStructure.shape[0] - 1):
            self.neuronalNetworkStructure[i].connectTo(self.neuronalNetworkStructure[i+1])
            pass
        
        pass

    def __setWeights(self):
        for i in range(self.neuronalNetworkStructure.shape[0] - 1):
            self.neuronalNetworkStructure[i].setWeights(generateRandomWeights=True)
            pass

        pass

    def __str__(self):
        outputNeuronalNetworkStructure = ""

        for i in range(self.neuronalNetworkStructure.shape[0]):
            outputNeuronalNetworkStructure += self.neuronalNetworkStructure[i].__str__() + "\n"
            if not isinstance(self.neuronalNetworkStructure[i].getLayerWeights(), type(None)):
                outputNeuronalNetworkStructure += str(self.neuronalNetworkStructure[i].getLayerWeights()) + "\n"
                pass
            pass
        
        return outputNeuronalNetworkStructure
        pass

    # Query neural network
    def forwarding(self, input):
        outputs = np.zeros((input.shape[0], self.neuronalNetworkStructure[-1].numberOfNeurons))

        if len(input.shape) != 2:
            raise ValueError("Als Eingabe wird ein 2Dim Array erwartet")
            pass
        elif input.shape[1] != self.neuronalNetworkStructure[0].numberOfNeurons:
            raise ValueError(f"Eingegebene Werte müsse der Anzahl an Neuronen entsprechen, hier: shape Array der Daten zum Formwarden {input.shape[1]}, Anzahl der InputNeuronen: {self.neuronalNetworkConnections[0].numberOfNeurons}")
            pass

        for queryData in range(input.shape[0]):
            self.neuronalNetworkStructure[0].setLayerInputs(input[queryData])

            for layer in range(self.neuronalNetworkStructure.size - 1):
                self.neuronalNetworkStructure[layer].setInputsNextLayer()
                self.neuronalNetworkStructure[layer + 1].getLayerNeuronsOutputValues()
                pass

            for outputneuron in range(self.neuronalNetworkStructure[-1].numberOfNeurons):
                outputs[queryData][outputneuron] = self.neuronalNetworkStructure[-1].getLayerNeuronsOutputValues()[outputneuron]
                pass
            pass

        return outputs
        pass

    pass
 
inputLayer = np.array([1, 2])
nHiddenLayer = np.array([[1,4],[2,3]])
outputLayer = np.array([2])

nn = neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)
print(nn.__str__())

trainData = ownFunctions.generateNTrainData(numberOfTrainData=2)
print(nn.forwarding(trainData))

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


