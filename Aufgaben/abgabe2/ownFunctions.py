__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.5"
__status__ = "Testing" # TODO test

# kernel imports
import numpy as np
import random

# own data imports
import constants
from constants import xMin, xMax, inputNeurons, invalidTrainDataMaxPoint, invalidTrainDataMinPoint, invalidTrainDataExklusivPointDistance, validDataValue, invalidDataValue


# Generates two random numbers from interval with gap
# returns valid values by default
def generateRandomNumbers(minIntervallStartValue=xMin, minIntervallEndValue=0, maxIntervallStartValue=0, maxIntervallEndValue=xMax, randomNumbers=1):
    generatedRandomNumbers = np.zeros(randomNumbers)

    numberOfRandomNumbersInMinInterval = int(randomNumbers / 2)
    numberOfRandomNumbersInMaxInterval = int(randomNumbers / 2)

    if randomNumbers % 2 != 0:
        rnd = random.randint(0, 1)
        if rnd == 0:
            numberOfRandomNumbersInMinInterval += 1
            pass
        else:   
            numberOfRandomNumbersInMaxInterval += 1
            pass
        pass

    for i in range(generatedRandomNumbers.size):
        if i < numberOfRandomNumbersInMinInterval:
            generatedRandomNumbers[i] = random.uniform(minIntervallStartValue, minIntervallEndValue)
            pass
        else:
            generatedRandomNumbers[i] = random.uniform(maxIntervallStartValue, maxIntervallEndValue)
            pass
        pass

    return generatedRandomNumbers
    pass

# Returns array with n TrainPoints(x,y)
def generateNTrainData(minIntervallStartValue=xMin, minIntervallEndValue=0, maxIntervallStartValue=xMax, maxIntervallEndValue=0, numberOfTrainData=1):
    trainData = np.zeros((numberOfTrainData, 2))

    for i in range(trainData.shape[0]):
        randomNumber = generateRandomNumbers(minIntervallStartValue, minIntervallEndValue, maxIntervallStartValue, maxIntervallEndValue, 2)
        trainData[i][0] = randomNumber[0]
        trainData[i][1] = randomNumber[1]
        pass

    return trainData
    pass

def generate_N_ValidAndInvalid_TrainData_Labeld_Shuffled(numberOfTrainData=1):
    trainDataShuffled = np.zeros((numberOfTrainData, 3))

    numberOfValidTrainDataLabeld = 0
    numberOfInvalidTrainDataLabeld = 0

    if numberOfTrainData % 2 == 0:
        numberOfValidTrainDataLabeld = int(numberOfTrainData / 2)
        numberOfInvalidTrainDataLabeld = int(numberOfTrainData / 2)
        pass
    else:
        numberOfValidTrainDataLabeld = int(numberOfTrainData / 2)
        numberOfInvalidTrainDataLabeld = int(numberOfTrainData / 2)

        # Randomness decides whether the valid or invalid training data has one more data set
        rnd = random.randint(0, 1)
        if rnd == 0:
            numberOfValidTrainDataLabeld += 1
            pass
        else:
            numberOfInvalidTrainDataLabeld += 1
            pass
        pass

    validTrainDataLabeld = generateNValidTrainDataLabeld(numberOfValidTrainDataLabeld)
    invalidTrainDataLabeld = generateNInvalidTrainDataLabeld(numberOfInvalidTrainDataLabeld)
    print(numberOfInvalidTrainDataLabeld)

    print("---valid:---\n" + str(validTrainDataLabeld) + "\n")
    print("---invalid:---\n" + str(invalidTrainDataLabeld) + "\n")
    for i in range(trainDataShuffled.shape[0]):
        if i < validTrainDataLabeld.shape[0]:
            trainDataShuffled[i] = validTrainDataLabeld[i]
            pass
        else:
            trainDataShuffled[i] = invalidTrainDataLabeld[i - validTrainDataLabeld.shape[0]]
            pass
        pass

    print(trainDataShuffled)

    pass

def generateNValidTrainData(numberOfValidTrainData=1):
    return generateNTrainData(xMin, 0, xMax, 0, numberOfValidTrainData)
    pass

# generates n invalid data
# 4 intervals must be considered: ]-1,1] ; [-1,1[ ; ]-1,0] ; [0,1[
def generateNInvalidTrainData(numberOfInvalidTrainData=1):
    invalidTrainData = np.zeros((numberOfInvalidTrainData, 2))
    intervals = np.array([[xMin - invalidTrainDataExklusivPointDistance, xMax],[xMin, xMax + invalidTrainDataExklusivPointDistance],[xMin - invalidTrainDataExklusivPointDistance, 0], [0, xMax + invalidTrainDataExklusivPointDistance]])

    print(intervals)

    pass

# labelvalue is on position -1
def generateNInvalidTrainDataLabeld(numberOfInvalidTrainData=1):
    return labelData(generateNInvalidTrainData(numberOfInvalidTrainData), invalidDataValue)
    pass

def generateNValidTrainDataLabeld(numberOfValidTrainData=1):
    return labelData(generateNValidTrainData(numberOfValidTrainData), validDataValue)
    pass

# labelvalue is on position -1
def labelData(dataToLabel, labelValue):
    dataLabeld = np.zeros((dataToLabel.shape[0], dataToLabel.shape[1] + 1))

    for i in range(dataLabeld.shape[0]):
        for j in range(dataLabeld.shape[1] - 1):
            dataLabeld[i][j] = dataToLabel[i][j]
            pass
        dataLabeld[i][-1] = labelValue
        pass

    return dataLabeld
    pass

# Only this function is called to generate random weights
def generateRandomWeights(startValue, endValue, numberOfWeights):
    return generateRandomWeights_NormalDistributionsCenter(startValue, endValue, numberOfWeights)
    pass

# distribution of the random weights random
def generateRandomWeights_standard(startValue, endValue, numberOfWeights):
    weights = np.zeros(numberOfWeights)

    for i in range(weights.shape[0]):
        weights[i] = np.random.uniform(startValue, endValue)
        pass
    return weights
    pass

# Equal distribution of the random weight around the mean
def generateRandomWeights_NormalDistributionsCenter(startValue, endValue, numberOfWeights):
    weights = np.zeros(numberOfWeights)
    distributionCenter = (startValue + endValue) / 2
    
    firstHalf = 0
    lastHalf = 0

    if numberOfWeights % 2 == 0:
        firstHalf = int(numberOfWeights / 2)
        lastHalf = int(numberOfWeights / 2)
        pass
    else:
        firstHalf = int(numberOfWeights / 2)
        lastHalf = int(numberOfWeights / 2 + 1)
        pass

    for i in range(firstHalf):
        weights[i] = np.random.uniform(distributionCenter, endValue)
        pass

    for i in range(lastHalf):
        weights[firstHalf + i] = np.random.uniform(startValue, distributionCenter)
        pass

    np.random.shuffle(weights)
    return weights
    pass


#print(generate_N_ValidAndInvalid_TrainData_Labeld_Shuffled(numberOfTrainData=5))
#print("\n")
#print(generateNInvalidTrainData(5))

print(generateRandomNumbers(randomNumbers=9))
