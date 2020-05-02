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

        # create inputlayer
        self.neuronalNetworkStructure[0] = nnl.neuronalNetworkLayer(inputLayerArray[0], inputLayerArray[1], inputLayerLabel, isInputLayer=True)

        # create hiddenLayer
        #for i in range(hiddenLayerNDIMArray.shape[0]):
        #    self.neuronalNetworkStructure[i + 1] = nnl.neuronalNetworkLayer(hiddenLayerNDIMArray[i][0], hiddenLayerNDIMArray[i][1], hiddenLayerLabel + " (" + str(i+1) + ")")
        #    pass

        # create outputLayer
        #self.neuronalNetworkStructure[-1] = nnl.neuronalNetworkLayer(0, outputLayerArray[0], outputLayerLabel, isOutputLayer=True)

        pass

    pass
 
inputLayer = np.array([1, 2])
nHiddenLayer = np.array([[1,4],[2,3]])
outputLayer = np.array([1])

#nn = neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)
nnl.neuronalNetworkLayer(1,2,"Input", isInputLayer=True)

## setup NN
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






