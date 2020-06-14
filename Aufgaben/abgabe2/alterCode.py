import math

sigmoid = lambda x: 1 / (1 + math.e**-x)

sigmoid_Ab_1 = lambda x: sigmoid(x) * (1 - sigmoid(x))

# Rechnungen

print(sigmoid_Ab_1(-0.2)*(0.055 * 0.3 + (-0.0868 * 0.9)))
#-----------------------------------------------------------------------------

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

                t1 = np.array(errorNextLayer * outputNextLayer * (1 - outputNextLayer), ndmin=2).T
                #print("t1: " + str(t1))
                t2 = np.array(outputThisLayer, ndmin=2).T
                #print((t1).shape)
                #print((t2).shape)
                t3 = np.dot(t1, t2.T)

                #print("t1 shape: " + str(t1.shape))
                #print("t2.T shape: " + str((t2).shape))
                #print("dot(t2.T,t1) shape: " + str(np.dot(t2.T,t1).shape))

                #TODO replace tmp_lr with constants.learningRate
                #print("Gewichte aktuell: \n" + str(self.neuronalNetworkStructure[-2 -i].getLayerWeights()))
                #print("Fehler vorheringe Schicht: \n" + str(self.neuronalNetworkStructure[-2 -i + 1].getLayerError()))
                tmp_lr = 0.1
                #print("Gewichtsänderung akt: \n" + str(tmp_lr * t3.T))
                #print("Alternative Gewichtsänderung: \n" + str(tmp_lr * np.dot(t2.T,t1)))

                #print(t3)
                #print("alt------------------")
                #print(self.neuronalNetworkStructure[-2 -i].getLayerWeights())
                #print("neu------------------")
                t4 = self.neuronalNetworkStructure[-2 -i].getLayerWeights() + tmp_lr * t3.T
                #print(t4)
                self.neuronalNetworkStructure[-2 -i].setWeights(useSpecificWeights=True, specificWeightsArray=t4)