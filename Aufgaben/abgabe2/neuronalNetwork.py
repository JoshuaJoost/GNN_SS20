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

# own data imports
import constants
from constants import inputNeurons, biasNeurons, hiddenNeurons, outputNeurons, activationSigmoid, learningRate
from constants import errorfunction
from constants import inputLayerLabel, hiddenLayerLabel, outputLayerLabel
import ownFunctions
import neuronalNetworkLayer as nnl
import ownTests

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

    # Query neural network
    def forwarding(self, input):
        outputs = None #np.zeros((input.shape[0], self.neuronalNetworkStructure[-1].numberOfNeurons))

        if len(input.shape) < 1 and len(input.shape) > 2:
            raise ValueError("Als Eingabe wird ein 1Dim oder 2Dim Array erwartet")
            pass

        # 1Dim Traindata
        if len(input.shape) == 1:
            outputs = np.zeros(self.neuronalNetworkStructure[-1].numberOfNeurons)

            if input.shape[0] < self.neuronalNetworkStructure[0].numberOfNeurons:
                errorMsg = "Eingegebene Werte müsse der Anzahl an Neuronen entsprechen, hier: shape Array der Daten zum Formwarden " + str(input.shape[1]) + ", Anzahl der InputNeuronen " + str(self.neuronalNetworkStructure[0].numberOfNeurons)
                raise ValueError(errorMsg)
                pass

            self.neuronalNetworkStructure[0].setLayerInputs(input[:self.neuronalNetworkStructure[0].numberOfNeurons])

            for layer in range(self.neuronalNetworkStructure.size - 1):
                self.neuronalNetworkStructure[layer].setInputsNextLayer()
                self.neuronalNetworkStructure[layer + 1].getLayerNeuronsAndBiasOutputValues()
                pass

            for outputneuron in range(self.neuronalNetworkStructure[-1].numberOfNeurons):
                outputs[outputneuron] = self.neuronalNetworkStructure[-1].getLayerNeuronsAndBiasOutputValues()[outputneuron]
                pass

            pass

        # 2Dim Traindata
        elif len(input.shape) == 2:
            outputs = np.zeros((input.shape[0], self.neuronalNetworkStructure[-1].numberOfNeurons))

            if input.shape[1] < self.neuronalNetworkStructure[0].numberOfNeurons:
                errorMsg = "Eingegebene Werte müsse der Anzahl an Neuronen entsprechen, hier: shape Array der Daten zum Formwarden " + str(input.shape[1]) + ", Anzahl der InputNeuronen " + str(self.neuronalNetworkStructure[0].numberOfNeurons)
                raise ValueError(errorMsg)
                pass

            for queryData in range(input.shape[0]):
                # The first n parameters are used as input parameters. All others (label parameters) are ignored.
                self.neuronalNetworkStructure[0].setLayerInputs(input[queryData][:self.neuronalNetworkStructure[0].numberOfNeurons])

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
            
            # calculate error
            # target value - output
            outputLayerError = np.reshape(errorfunction(labeldTrainData[trainData][-1], output),(1,1))
            self.neuronalNetworkStructure[-1].setLayerError(outputLayerError)
            #print(self.neuronalNetworkStructure[-1].getLayerError())

            # backpropagation
            for i in range(self.neuronalNetworkStructure.shape[0] - 1):
                ##--- Calculate the new weights depending on the error of the following layer
                # Set the errors of the layers
                #print("----" + str(i) + "----")
                #print(self.neuronalNetworkStructure[-2 -i].getLayerName())
                #print("prev layer error shape: " + str(self.neuronalNetworkStructure[-2 -i + 1].getLayerError().shape))
                #print("weights shape: " + str(self.neuronalNetworkStructure[-2 -i].getLayerWeights().shape))
                        
                layerError = np.dot(self.neuronalNetworkStructure[-2 -i].getLayerWeights(), self.neuronalNetworkStructure[-2 -i +1].getLayerError())
                #print("layer Error shape: " + str(layerError.shape))
                
                ##--- set layer error for backpropagation
                # inputlayer does not propagate an error back
                #print(self.neuronalNetworkStructure[-2 -i].getIsInputLayer())
                if not self.neuronalNetworkStructure[-2 -i].getIsInputLayer():
                    layerErrorBackpropagation = np.zeros((self.neuronalNetworkStructure[-2 -i].getNumberOfNeurons(), 1))
                    for j in range(layerErrorBackpropagation.shape[0]):
                        layerErrorBackpropagation[j] = layerError[j + self.neuronalNetworkStructure[-2 -i].getNumberOfBiasNeurons()]
                        pass
                    #print("layer Error Backprop shape: " + str(layerErrorBackpropagation.shape))
                    self.neuronalNetworkStructure[-2 -i].setLayerError(layerErrorBackpropagation)
                    pass

                ##--- Adjusting the weights
                errorNextLayer = np.reshape(self.neuronalNetworkStructure[-2 -i +1].getLayerError(), (-1))
                outputNextLayer = self.neuronalNetworkStructure[-2 -i +1].getLayerNeuronsOutputValues()
                outputThisLayer = self.neuronalNetworkStructure[-2 -i].getLayerNeuronsAndBiasOutputValues()

                t1 = np.array(errorNextLayer * outputNextLayer * (1 - outputNextLayer), ndmin=2)
                t2 = np.array(outputThisLayer, ndmin=2)
                #print((t1).shape)
                #print((t2).shape)
                t3 = np.dot(t1.T, t2)
                #print(t3)
                #print("alt------------------")
                #print(self.neuronalNetworkStructure[-2 -i].getLayerWeights())
                #print("neu------------------")
                t4 = self.neuronalNetworkStructure[-2 -i].getLayerWeights() + constants.learningRate * t3.T
                #print(t4)
                self.neuronalNetworkStructure[-2 -i].setWeights(useSpecificWeights=True, specificWeightsArray=t4)
                
                pass

            pass
        pass

    pass
 
inputLayer = np.array([1, 2])
nHiddenLayer = np.array([[1,4]])
outputLayer = np.array([1])

nn = neuronalNetwork(inputLayer, nHiddenLayer, outputLayer)
#print(nn.__str__())

trainData = ownFunctions.trainDataLabeld_shuffeld(5000000)
validData = ownFunctions.validDataLabeld(10000)
invalidData = ownFunctions.invalidDataLabeld(10000)

testDataShuffeld = ownFunctions.trainDataLabeld_shuffeld(1000)
testDataValid = ownFunctions.validDataLabeld(20)
testDataInvalid = ownFunctions.invalidDataLabeld(20)

nn.trainWithlabeldData(trainData)
outputs = np.reshape(nn.forwarding(testDataShuffeld), (testDataShuffeld.shape[0]))

targetValues = np.zeros((outputs.shape[0]))
for i in range(targetValues.shape[0]):
    targetValues[i] = testDataShuffeld[i][2]
    pass
print(ownTests.evaluatesTrainingCycle(targetValues, outputs))


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


