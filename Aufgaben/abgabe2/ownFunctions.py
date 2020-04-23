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
from constants import xMin, xMax, inputNeurons, invalidTrainDataMaxPoint, invalidTrainDataMinPoint, invalidTrainDataExklusivPointDistance


# Generates two random numbers from interval with gap
# returns valid values by default
def generateRandomNumbers(minIntervallStartValue=xMin, minIntervallEndValue=0, maxIntervallStartValue=xMax, maxIntervallEndValue=-0, numberOfValues=inputNeurons):
    randomGenerateNumbers = np.zeros(numberOfValues) 

    # contains <numberOfValues * 2> negativ values and <numberOfValues * 2> positive numbers
    randomGeneratedNumbersPool = np.zeros(numberOfValues * 2)

    for i in range(randomGeneratedNumbersPool.size):
        # generate negative numbers
        if i < randomGeneratedNumbersPool.size / 2:
            randomGeneratedNumbersPool[i] = random.uniform(minIntervallStartValue, minIntervallEndValue)
            pass
        # generate positive numbers
        else:
            randomGeneratedNumbersPool[i] = random.uniform(maxIntervallStartValue, maxIntervallEndValue)
            pass
        pass

    for i in range(randomGenerateNumbers.size):
        valueChoosed = random.choice(randomGeneratedNumbersPool)
        randomGenerateNumbers[i] = valueChoosed

        # Prevent duplication of values
        randomGeneratedNumbersPool = np.delete(randomGeneratedNumbersPool, np.where(randomGeneratedNumbersPool == valueChoosed)[0][0])
        pass

    return randomGenerateNumbers
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

def generateNValidTrainData(numberOfValidTrainData=1):
    return generateNTrainData(xMin, 0, xMax, 0, numberOfValidTrainData)
    pass

# generates n invalid data
# 4 intervals must be considered: ]-1,1] ; [-1,1[ ; ]-1,0] ; [0,1[
def generateNInvalidTrainData(numberOfInvalidTrainData=1):
    toReturn_InvalidTrainData = np.zeros((numberOfInvalidTrainData, 2))
    intervals = np.array([[xMin - invalidTrainDataExklusivPointDistance, xMax],[xMin, xMax + invalidTrainDataExklusivPointDistance],[xMin - invalidTrainDataExklusivPointDistance, 0], [0, xMax + invalidTrainDataExklusivPointDistance]])
    
    numberOfInvalidTrainDataExtended = numberOfInvalidTrainData
    while numberOfInvalidTrainDataExtended % intervals.shape[0] != 0:
        numberOfInvalidTrainDataExtended += 1
        pass

    # generates dynamic intervals.shape[0] interval variables and generates numberOfInvalidTrainDataExtended / intervals.shape[0] points of this interval
    # i.e.: by 4 intervals and 12 requested data it generates 4 interval variables containing 3 points of the intervall (each of them)
    trainingDataViaInterval = {}
    for i in range(intervals.shape[0]):
        trainingDataViaInterval["interval%s" %i] = generateNTrainData(intervals[i][0], invalidTrainDataMinPoint, intervals[i][1], invalidTrainDataMaxPoint, int(numberOfInvalidTrainDataExtended / intervals.shape[0]))
        pass

    # Merge the data of the individual intervals
    appendedTrainData = np.zeros((trainingDataViaInterval["interval0"].shape[0] * intervals.shape[0],2))
    for iVal in range(int(toReturn_InvalidTrainData.shape[0] / trainingDataViaInterval["interval0"].shape[0])):
        for iInterval in range(int(toReturn_InvalidTrainData.shape[0] / intervals.shape[0])):
            appendedTrainData[iVal + iInterval + (iVal * trainingDataViaInterval["interval%s" %iVal].shape[0] - iVal)] = trainingDataViaInterval["interval%s" %iVal][iInterval]
            pass
        pass

    # Return only the number of the requested training data
    for i in range(numberOfInvalidTrainData):
        toReturn_InvalidTrainData[i] = appendedTrainData[i]
        pass

    return toReturn_InvalidTrainData
    pass









