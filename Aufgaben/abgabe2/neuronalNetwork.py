__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"
##--- TODO
# - [Joshua] Backpropagation
# - testen
# - [optional]: importieren und exportieren des Neuronalen Netzes (um es speichern und laden zu können)

# kernel imports
import numpy as np
import scipy.special 
import types
import random

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
        for layer in range(self.neuronalNetworkStructure.size):
            # set values of input layer
            if self.neuronalNetworkStructure[layer].getIsInputLayer():
                # target value is not considered

                if input.shape[0] == 2:
                    # input: shape [x, y]
                    self.neuronalNetworkStructure[layer].setLayerInputs(input[:])
                    pass
                elif input.shape[0] == 3:
                    # input: shape [x, y, targetValue]
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
            #error = labeldTrainData[trainData][2] - output
            #print(str(error) + "-" + str(output) + " = (error) " + str(error-output))
            
            # calculate error
            # target value - output
            #outputLayerError = np.reshape(errorfunction(labeldTrainData[trainData][-1], output),(1,1))
            #self.neuronalNetworkStructure[-1].setLayerError(outputLayerError)
            #print(self.neuronalNetworkStructure[-1].getLayerError())

            # backpropagation
            # calculate and set delta value
            for i in range(self.neuronalNetworkStructure.shape[0] - 1):
                # output layer
                if i == 0:
                    for outputNeuronI in range(self.neuronalNetworkStructure[-1 - i].getNumberOfNeurons()):
                        networkInputOutputneuronI = self.neuronalNetworkStructure[-1 - i].getLayerNeurons()[outputNeuronI].getInput()
                        deltaOutputNeuronI = activationFunctionDerived_1(networkInputOutputneuronI) * (labeldTrainData[trainData][2] - output[outputNeuronI])
                        
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
                #print("old weights -------------------")
                #print(self.neuronalNetworkStructure[-1 - i - 1].getLayerWeights())
                #print("New Weights ------------------")
                #print("delta weights: " + str(deltaWeights.T.shape))
                #print(str(deltaWeights.T))

                # calculate and set new weights
                #print("old weights: " + str(self.neuronalNetworkStructure[-1 - i -1].getLayerWeights()))
                newWeights = self.neuronalNetworkStructure[-1 - i -1].getLayerWeights() + deltaWeights.T
                self.neuronalNetworkStructure[-1 - i -1].setWeights(useSpecificWeights = True, specificWeightsArray = newWeights)
                #print("new weights: " + str(newWeights))
                pass
            pass
        pass

    pass
 
inputLayer = np.array([1, 2])
nHiddenLayer = np.array([[1,4]])
outputLayer = np.array([1])

nn = neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)

trainData = ownFunctions.getRandomTrainData(1)

for i in range(trainData.shape[0]):
    print(nn.forwarding(trainData[i]))
    pass



## -- Training phase
#trainData = ownFunctions.getRandomTrainData(200000)
#print(trainData)
#np.random.shuffle(trainData)
#print(trainData)
#trainDataValid = ownFunctions.validDataLabeld(10000)
#trainDataInvalid = ownFunctions.invalidDataLabeld(10000)

#print("Trainingsphase 1")
#nn.trainWithlabeldData(trainData)
#print("Trainingsphase 2")
#nn.trainWithlabeldData(trainDataValid)
#print("Trainingsphase 3")
#nn.trainWithlabeldData(trainData)
#print("Trainingsphase 4")
#nn.trainWithlabeldData(trainData)
#print("Trainingsphase 5")
#nn.trainWithlabeldData(trainData)
#print("Training beendet")

## -- Test phase
#testData = trainData # Indexpositions: 0 = x, 1 = y, 2 = targetValue
#testData = ownFunctions.trainDataLabeld_shuffeld(100) #np.zeros([100, 3])

#tmpTrainData = trainData
#for i in range(testData.shape[0]):
#    index = int(random.uniform(0, tmpTrainData.shape[0] - 1))

#    testData[i] = tmpTrainData[index]

#    tmpTrainData = np.delete(tmpTrainData, index, 0)
#    pass

#outputsForwarding = nn.forwarding(testData)

## -- Evaluation phase
#okValue = 0

#print("Output --- Target")
#for i in range(outputsForwarding.size):
#    if abs(testData[i][2] - outputsForwarding[i]) >= 0 and abs(testData[i][2] - outputsForwarding[i]) <= 0.2:
#        print("\x1B[32m" + str(outputsForwarding[i]) + " --- " + str(testData[i][2]) + "\x1B[0m")
#        okValue += 1
#        pass
#    else:
#        print("\x1B[31m" + str(outputsForwarding[i]) + " --- " + str(testData[i][2]) + "\x1B[0m")
#        pass

#    pass
#print(str(int(okValue * testData.shape[0] / 100)) + '% richtig')

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
#print(data)
#print(nn.__str__())

#trainData = ownFunctions.trainDataLabeld_shuffeld(100000)
#validData = ownFunctions.validDataLabeld(10000)
#invalidData = ownFunctions.invalidDataLabeld(10000)

#testDataShuffeld = ownFunctions.trainDataLabeld_shuffeld(100)
#testDataValid = ownFunctions.validDataLabeld(20)
#testDataInvalid = ownFunctions.invalidDataLabeld(20)

#nn.trainWithlabeldData(trainData)

#tquery = lambda x,y: 0.8 if x**2 + y**2 <= 1 else 0.0
#view.printCircle(2, query = nn.forwarding)
#outputsForwarding = nn.forwarding(testDataShuffeld)

#outputs = np.reshape(nn.forwarding(testDataShuffeld), (testDataShuffeld.shape[0]))

#targetValues = np.zeros((outputsForwarding.shape[0]))
#for i in range(targetValues.shape[0]):
#    targetValues[i] = trainData[i][2] # testDataShuffeld[i][2]
#    pass
#print("targets\n" + str(targetValues) + "\n")
#print("outputs\n" + str(outputsForwarding) + "\n")
#print(ownTests.evaluatesTrainingCycle(targetValues, outputsForwarding))


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


