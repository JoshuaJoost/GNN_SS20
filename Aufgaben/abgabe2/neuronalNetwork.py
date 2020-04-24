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
        inputs = np.array(inputsArray).T

        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)

        outputLayerInputs = np.dot(self.who, hiddenOutputs)
        finalOutput = self.activation_function(outputLayerInputs)

        return finalOutput
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

        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)
        resultsInputs = np.dot(self.who, hiddenOutputs)
        resultsOutputs = self.activation_function(resultsInputs)

        outputErrors = targets - resultsOutputs
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
        self.wih += self.lr * np.dot((hiddenErrors * hiddenOutputs * (1 - hiddenOutputs)), np.transpose(inputs))

        pass

    def getWIH(self):
        return self.wih
        pass

    def getOutputErrors(self):
        return self.outputErrors
        pass

    pass
 


nn = neuralNetwork()
#print(nn.getWIH())
for i in range(1000):
    #data = ownFunctions.generateNValidTrainDataLabeld(1)
    if i % 2 == 0:
        data = ownFunctions.generateNValidTrainDataLabeld(1)
        pass
    else:
        data = ownFunctions.generateNInvalidTrainDataLabeld(1)
        pass
    nn.trainWithLabeldData(data)
    pass

#for i in range(1000):
    #data = ownFunctions.generateNValidTrainDataLabeld(1)
#    nn.trainWithLabeldData(ownFunctions.generateNInvalidTrainDataLabeld(1))
#    pass
#print(nn.getWIH())

#print(nn.getOutputErrors())
print("test valid (0.8)")
print(nn.query(ownFunctions.generateNValidTrainData(1)))
print(nn.query(ownFunctions.generateNValidTrainData(1)))
print(nn.query(ownFunctions.generateNValidTrainData(1)))
print("test invalid (0.0)")
print(nn.query(ownFunctions.generateNInvalidTrainData(1)))
print(nn.query(ownFunctions.generateNInvalidTrainData(1)))
print(nn.query(ownFunctions.generateNInvalidTrainData(1)))






