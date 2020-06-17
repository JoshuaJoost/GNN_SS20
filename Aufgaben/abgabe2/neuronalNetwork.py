__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "1.0"
__status__ = "Ready"
##--- TODO
# - [optional]: importieren und exportieren des Neuronalen Netzes (um es speichern und laden zu können)

# kernel imports
import numpy as np
import scipy.special 
import types
import random
import math

# own data imports
import constants
from constants import inputNeurons, biasNeurons, hiddenNeurons, outputNeurons, activationFunction, activationFunctionDerived_1, learningRate
from constants import errorfunction
from constants import inputLayerLabel, hiddenLayerLabel, outputLayerLabel
import ownFunctions
import neuronalNetworkLayer as nnl
import ownTests
import view

class neuronalNetwork:    
    # :param2: inputLayerArray: shape(1,numberOfInputNeurons) [0] = BiasNeurons, [1] = InputNeurons
    # :param3: hiddenLayerNDIMArray: shape(numberOfHiddenLayers, 2) [x][0] = NumberOfBiasNeurons, [x][1] = NumberOfNeurons
    # :param4: outputLayerArray: shape(numberOfOutputNeurons) [0] = NumberOfOutputNeurons
    def __init__(self, inputLayerArray, hiddenLayerNDIMArray, outputLayerArray):
        # object variables
        self.errorValues = np.empty(shape=1) # set in backpropagation process
        self.errorValues = np.delete(self.errorValues, 0)

        ## --- Generate and connect layer
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
        self.__initialiseWeights()

        pass

    def __connectLayers(self):
        for i in range(self.neuronalNetworkStructure.shape[0] - 1):
            self.neuronalNetworkStructure[i].connectTo(self.neuronalNetworkStructure[i+1])
            pass
        
        pass

    def __initialiseWeights(self):
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

    # forwarding function neuronal network
    # :param input: type = np.array, shape = 3 [x, y, targetValue] or shape = 2 [x, y]
    def forwarding(self, input): 
        #print(input) 
        for layer in range(self.neuronalNetworkStructure.size):
            # set values of input layer
            if self.neuronalNetworkStructure[layer].getIsInputLayer():
                if input.shape[0] == 2:
                    # input: shape [x, y]
                    self.neuronalNetworkStructure[layer].setLayerInputs(input[:])
                    pass
                elif input.shape[0] == 3:
                    # input: shape [x, y, targetValue]
                    # target value is not considered
                    self.neuronalNetworkStructure[layer].setLayerInputs(input[:-1])
                    pass
                else:
                    raise ValueError("Der forwarding Funktion muss ein Array des Shape 2 (x,y) oder 3 (x,y,targetValue) übergeben werden. Übergebener shape: " + str(input.shape[0]))
                    pass
                pass
            # set values of hidden and output layer (in the same way)
            else:
                self.neuronalNetworkStructure[layer].setLayerInputs(np.dot(self.neuronalNetworkStructure[layer - 1].getLayerNeuronsAndBiasOutputValues().T, self.neuronalNetworkStructure[layer - 1].getLayerWeights())[0])
                pass
            pass

        return self.neuronalNetworkStructure[-1].getLayerNeuronsAndBiasOutputValues()
        pass

    # :param2: labeldTrainData: Data must have the shape (numberOfTrainingData, numberOfInputValues + 1), numberOfInputValues = numberOfInputNeurons
    def trainWithlabeldData(self, labeldTrainData):

        if len(labeldTrainData.shape) != 2:
            raise ValueError("Als Eingabe wird ein 2Dim Array erwartet")
            pass
        elif labeldTrainData.shape[1] < self.neuronalNetworkStructure[0].numberOfNeurons + 1: # +1 because of the label
            errorMsg = "Eingegebene Werte müsse der Anzahl an Neuronen (+1 für das Label) entsprechen, hier: shape Array der Daten zum Formwarden " + str(input.shape[1]) + ", Anzahl der InputNeuronen " + str(self.neuronalNetworkStructure[0].numberOfNeurons)
            raise ValueError(errorMsg)
            pass

        for trainData in range(labeldTrainData.shape[0]):
            # forwarding
            output = self.forwarding(labeldTrainData[trainData])

            # backpropagation
            # calculate and set delta value
            for i in range(self.neuronalNetworkStructure.shape[0] - 1):
                # output layer
                if i == 0:
                    for outputNeuronI in range(self.neuronalNetworkStructure[-1 - i].getNumberOfNeurons()):
                        networkInputOutputneuronI = self.neuronalNetworkStructure[-1 - i].getLayerNeurons()[outputNeuronI].getInput()
                        
                        # calc error
                        error = labeldTrainData[trainData][2] - output[outputNeuronI]

                        # save error
                        self.errorValues = np.append(self.errorValues, error)

                        # calc delta value
                        deltaOutputNeuronI = activationFunctionDerived_1(networkInputOutputneuronI) * error
                        
                        # set delta value
                        self.neuronalNetworkStructure[-1 - i].getLayerNeurons()[outputNeuronI].setDelta(deltaOutputNeuronI)
                        pass
                    pass 
                # hidden layer
                else:
                    for neuron in range(self.neuronalNetworkStructure[-1 -i].getLayerNeurons().size - self.neuronalNetworkStructure[-1 -i].getNumberOfBiasNeurons()):
                        networkInputHiddenneuronI = self.neuronalNetworkStructure[-1 - i].getLayerNeurons()[neuron + self.neuronalNetworkStructure[-1 - i].getNumberOfBiasNeurons()].getInput()
                        deltaHiddenNeuronI = activationFunctionDerived_1(networkInputHiddenneuronI) * (np.dot(self.neuronalNetworkStructure[-1 - i].getLayerWeights()[neuron + self.neuronalNetworkStructure[-1 - i].getNumberOfBiasNeurons()],self.neuronalNetworkStructure[-1 - i + 1].getLayerDeltavalueMatrix()))
                        
                        # set delta value
                        self.neuronalNetworkStructure[-1 - i].getLayerNeurons()[neuron + self.neuronalNetworkStructure[-1 - i].getNumberOfBiasNeurons()].setDelta(deltaHiddenNeuronI)
                        pass                    
                    pass              
                pass

            # calculate and set new weights
            for i in range(self.neuronalNetworkStructure.shape[0] - 1):
                # calculate the delta value of the weights
                deltaWeights = learningRate * (np.dot(self.neuronalNetworkStructure[-1 - i].getLayerDeltavalueMatrix(), self.neuronalNetworkStructure[-1 - i - 1].getLayerNeuronsAndBiasOutputValues().T))
                newWeights = self.neuronalNetworkStructure[-1 - i -1].getLayerWeights() + deltaWeights.T
                self.neuronalNetworkStructure[-1 - i -1].setWeights(useSpecificWeights = True, specificWeightsArray = newWeights)
                pass
            pass
        pass

    def preparePlotData_Error(self, dataDivisor = 1000):
        numberOfData = int(self.errorValues.size / dataDivisor)

        if numberOfData == 0 or self.errorValues.size % dataDivisor > 0:
            numberOfData += 1
            pass

        plotData = np.zeros([numberOfData])

        elementTranslation = 0
        for i in range(plotData.size):
            startIndexPos_ErrorGroup = i * dataDivisor + elementTranslation
            endIndexPos_ErrorGroup = (i + 1) * dataDivisor

            if i+1 == plotData.size:
                endIndexPos_ErrorGroup = self.errorValues.size
                pass
            
            plotData[i] = np.median(self.errorValues[startIndexPos_ErrorGroup:endIndexPos_ErrorGroup])

            if math.isnan(plotData[i]):
                plotData[i] = self.errorValues[-1]
                pass

            elementTranslation = 1
            pass

        return plotData
        pass

    pass