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
from constants import inputNeurons, biasNeurons, hiddenNeurons, outputNeurons, activationFunc, learningRate
from constants import INPUT_LAYER, HIDDEN_LAYER
import ownFunctions

### TODO Implement class of the neural network
### The main file calls this

class neuralNetwork:
    outputErrors = None

    # Generates 3-layer I-H-O neuronal network
    def __init__(self, inputNodes=inputNeurons, hiddenNodes=hiddenNeurons, outputNodes=outputNeurons, biasNeuronPerNode=biasNeurons, activationFunction=activationFunc, alphaLearningRate=learningRate):
        # Sets the neurons per layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # link weight matrices, wih and who
        # Weights are linked from node i to node j
        ## Structure of matrix
        # - [w11 w21]
        # - [w12 w22] etc
        self.wih = np.random.normal(0.0, pow(self.hNodes, 0.5), (self.hNodes, self.iNodes))
        self.who = np.random.normal(0.0, pow(self.oNodes, 0.5), (self.oNodes, self.hNodes))

        # learning rate alpha
        self.lr = alphaLearningRate

        # activation function
        self.activation_function = activationFunction

        pass

    def query(self, inputsArray):
        outputs = np.zeros((inputsArray.shape[0]))
        
        for i in range(inputsArray.shape[0]):
            inputs = np.array(inputsArray).T

            hiddenInputs = np.dot(self.wih, inputs)
            hiddenOutputs = self.activation_function(hiddenInputs)

            outputLayerInputs = np.dot(self.who, hiddenOutputs)
            outputs[i] = self.activation_function(outputLayerInputs)[0][i]
            pass

        return outputs
        pass

    # labels have to be on last index position
    def trainWithLabeldData(self, labeldInputsArray):
        inputs = np.zeros((labeldInputsArray.shape[0], labeldInputsArray.shape[1] - 1))
        targets = np.zeros((labeldInputsArray.shape[0], 1))
        #errors = np.zeros((1, targets.shape[0]))

        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                inputs[i][j] = labeldInputsArray[i][j]
                pass
            pass

        for i in range(targets.shape[0]):
            targets[i] = labeldInputsArray[i][-1]
            pass

        inputs = inputs.T
        targets = targets.T

        for i in range(labeldInputsArray.shape[0] - 1):
            hiddenInputs = np.dot(self.wih, inputs[0][i])
            hiddenOutputs = self.activation_function(hiddenInputs)
            resultsInputs = np.dot(self.who, hiddenOutputs)
            resultsOutputs = self.activation_function(resultsInputs)

            outputErrors = targets[0][i] - resultsOutputs
            #outputErrors = ((targets - resultsOutputs)**2).mean(axis=None)

            if isinstance(self.outputErrors, type(None)):
                self.outputErrors = abs(outputErrors)
                pass
            else: 
                self.outputErrors = np.append(self.outputErrors, abs(outputErrors))
                pass
            
            hiddenErrors = np.dot(self.who.T, outputErrors)

            # update weights
            self.who += self.lr * np.dot((outputErrors * resultsOutputs * (1 - resultsOutputs)), np.transpose(hiddenOutputs))
            self.wih += self.lr * np.dot((hiddenErrors * hiddenOutputs * (1 - hiddenOutputs)), np.transpose(inputs[0][i]))
            pass

        pass

    def getWIH(self):
        return self.wih
        pass

    def getWIH_Labeld(self):
        return self.getLayerWeightsLabeld(self.wih, "InputNeuron", INPUT_LAYER)
        pass

    def getWHO(self):
        return self.who
        pass

    def getWHO_Labeld(self):
        return self.getLayerWeightsLabeld(self.who, "HiddenNeuron", HIDDEN_LAYER)
        pass

    def getAllWeights_Labeld(self):
        nnWeightsLabeld = str(self.getWIH_Labeld())
        nnWeightsLabeld += "\n"
        nnWeightsLabeld += str(self.getWHO_Labeld())

        return nnWeightsLabeld
        pass

    def getLayerWeightsLabeld(self, weights, neuronNameOfActualLayer, postLayerName):
        labeldWidthsString = postLayerName
        
        for actualLayerNeuron in range(weights.shape[1]):
            labeldWidthsString += "\n" + neuronNameOfActualLayer + str(actualLayerNeuron + 1)
            for postLayerNeuron in range(weights.shape[0]):
                labeldWidthsString += "\nw" + str(postLayerNeuron + 1) + str(actualLayerNeuron + 1) + ": " + str(weights[postLayerNeuron][actualLayerNeuron])
                pass
            labeldWidthsString += "\n"
            pass

        return labeldWidthsString
        pass

    def getOutputErrors(self):
        return self.outputErrors
        pass

    pass
 
nn = neuralNetwork()
print(nn.getAllWeights_Labeld())
#nn.trainWithLabeldData(ownFunctions.generateNInvalidTrainDataLabeld(2000))
#print(nn.query(ownFunctions.generateNInvalidTrainData(5)))
#print(nn.getWIH())
#nn.trainWithLabeldData(ownFunctions.generateNValidTrainDataLabeld(2000))
#print(nn.query(ownFunctions.generateNValidTrainData(5)))
#print(nn.getWIH())







