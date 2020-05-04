__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.5"
__status__ = "Testing"
##--- TODO 
# - [Joshua] generate invalid train data
# - [Joshua] generate valid and invalid train data shuffled
# - testen

# kernel imports
import numpy as np
import random
import math

# own data imports
import constants
from constants import xMin, xMax, inputNeurons, invalidTrainDataMaxPoint, invalidTrainDataMinPoint, invalidTrainDataExklusivPointDistance, validDataValue, invalidDataValue
from ownTests import checkWhetherPointsLie_Outside_TheUnitCircle, checkWhetherPointsLie_Inside_TheUnitCircle, pointsLiesOnUniCircleEdge

# Tested with pointsLiesOnUniCircleEdge, errors are in the range e-16, Method OK
def borderOfUnitCircle(points=1):
    pointsBorderUnitCircle = np.zeros((points, 2))
    
    alpha = np.arange(0, 360, 360/points)
    for i in range(pointsBorderUnitCircle.shape[0]):
        pointsBorderUnitCircle[i][0] = math.sin(math.radians(alpha[i]))
        pointsBorderUnitCircle[i][1] = math.cos(math.radians(alpha[i]))
        pass

    return pointsBorderUnitCircle
    pass

# Tested with checkWhetherPointsLie_Inside_TheUnitCircle, No Errors, Method OK
def withinUnitCircle(points=1):
    pointsWithinUnitCircle = np.zeros((points, 2))

    # Intervals to divide the circle over degrees
    # [0, (math.pi / 2) - 0.001],
    # [math.pi / 2, math.pi - 0.001],
    # [math.pi, (3*math.pi / 2) - 0.001],
    # [3*math.pi / 2, 0 - 0.001]])
    # converted into radians for uniformity to simplify the interval
    intervalsDegrees = np.array([ 
        [0, 90 - 0.001],  
        [90, 180 - 0.001], 
        [180, 270 - 0.001], 
        [270, 360 - 0.001]])

    # Intervals to divide the circle over radius
    # index0 <= r < index 1
    intervalRadius = np.array([
        [0, 1/3 - 0.001],
        [1/3, 2/3 - 0.001],
        [2/3, 1 - 0.001]
    ])

    totalIntervals = np.zeros((intervalsDegrees.shape[0] * intervalRadius.shape[0], intervalsDegrees.shape[1], intervalRadius.shape[1]))
    
    for degree in range(intervalsDegrees.shape[0]):
        for radius in range(intervalRadius.shape[0]):
            totalIntervals[intervalRadius.shape[0] * degree + radius][0] = intervalsDegrees[degree]
            totalIntervals[intervalRadius.shape[0] * degree + radius][1] = intervalRadius[radius]
            pass
        pass

    # one point in each interval
    if points <= totalIntervals.shape[0]:     
        # Random selection of intervals depending on the number of points
        # intervalChoosed shape: [0] = number of points, [1] = Number of intervals (degree and radius interval), [2] = number of elements in intervals
        intervalChoosed = np.zeros((points, 2, 2))

        for interval in range(intervalChoosed.shape[0]):
            # TODO random.randint(0, totalIntervals.shape[0] - 1) um index zu bekommen, folgenden 2 zeilen - 1 rausnehmen
            choosedInterval = random.randint(0, totalIntervals.shape[0])
            intervalChoosed[interval] = totalIntervals[choosedInterval - 1]
            totalIntervals = np.delete(totalIntervals, choosedInterval - 1, 0)
            pass
        pass

        #-- generate x,y points within choosed intervals
        # [i][0][0,1] degrees
        # [i][1][0,1] radii
        for i in range(pointsWithinUnitCircle.shape[0]):
            alpha = random.uniform(intervalChoosed[i][0][0], intervalChoosed[i][0][1])
            radius = random.uniform(intervalChoosed[i][1][0], intervalChoosed[i][1][1])
            # x coordinate
            pointsWithinUnitCircle[i][0] = math.cos(math.radians(alpha)) * radius
            # y coordinate
            pointsWithinUnitCircle[i][1] = math.sin(math.radians(alpha)) * radius
            pass
        
    else:
        intervalChoosed = totalIntervals
        
        # Generation of equally distributed points within the intervals
        for i in range(int(points / intervalChoosed.shape[0])):
            for j in range(intervalChoosed.shape[0]):
                alpha = random.uniform(intervalChoosed[j][0][0], intervalChoosed[j][0][1])
                radius = random.uniform(intervalChoosed[j][1][0], intervalChoosed[j][1][1])
                pointsWithinUnitCircle[j + i * intervalChoosed.shape[0]][0] = math.cos(math.radians(alpha)) * radius
                pointsWithinUnitCircle[j + i * intervalChoosed.shape[0]][1] = math.sin(math.radians(alpha)) * radius
                pass
            pass

        randomlyEvenlyDistributedIntervals = np.zeros((points % intervalChoosed.shape[0], 2, 2)) 

        for i in range(randomlyEvenlyDistributedIntervals.shape[0]):
            choosedInterval = random.randint(0, totalIntervals.shape[0])
            randomlyEvenlyDistributedIntervals[i] = totalIntervals[choosedInterval - 1]
            totalIntervals = np.delete(totalIntervals, choosedInterval - 1, 0)
            pass
        pass

        # points in random, but equally distributed, intervals
        for i in range(randomlyEvenlyDistributedIntervals.shape[0]):
            alpha = random.uniform(randomlyEvenlyDistributedIntervals[i][0][0], randomlyEvenlyDistributedIntervals[i][0][1])
            radius = random.uniform(randomlyEvenlyDistributedIntervals[i][1][0], randomlyEvenlyDistributedIntervals[i][1][1])
            pointsWithinUnitCircle[-1 - i][0] = math.cos(math.radians(alpha)) * radius
            pointsWithinUnitCircle[-1 - i][1] = math.sin(math.radians(alpha)) * radius
            pass
        pass

    return pointsWithinUnitCircle
    pass

def outsideUnitCircle(points=1):
    pointsOutsideUnitCircle = np.zeros((points, 2))
    

    



    pointsOutsideUnitCircle
    pass

def pointsCloseToUniCircleBorder(points=1):
    points_ret = np.zeros((points, 2))
    radiusCloseToUnitCircle = np.array([1.1, 1.2])

    ##--- Points close to the unit circle
    # 4 Intervalls around the unit circle
    # [0, 90[; [90,180[; [180,270[; [270,360[ -> numbers in degree
    intervallsDegreeNearUnitCircle = np.array([
        [0, 90 - 0.001],
        [90, 180 - 0.001],
        [180, 270 - 0.001],
        [270, 360 - 0.001]
    ])

    intervallsNearUnitCircleChoosed = None
    if points < intervallsDegreeNearUnitCircle.shape[0]:
        intervallsNearUnitCircleChoosed = np.zeros([points, intervallsDegreeNearUnitCircle.shape[1]])

        # choose random intervalls
        for i in range(intervallsNearUnitCircleChoosed.shape[0]):
            choosedIntervall = random.randint(0, intervallsDegreeNearUnitCircle.shape[0] - 1)
            intervallsNearUnitCircleChoosed[i] = intervallsDegreeNearUnitCircle[choosedIntervall]
            intervallsDegreeNearUnitCircle = np.delete(intervallsDegreeNearUnitCircle, choosedIntervall, 0)
            pass
        pass
    else:
        intervallsNearUnitCircleChoosed = intervallsDegreeNearUnitCircle
        pass

    rndIntervalls = points % intervallsNearUnitCircleChoosed.shape[0]

    for i in range(int(points / intervallsNearUnitCircleChoosed.shape[0])):
        for j in range(intervallsNearUnitCircleChoosed.shape[0]):
            points_ret[j + i * intervallsNearUnitCircleChoosed.shape[0]][0] = math.cos(math.radians(random.uniform(intervallsNearUnitCircleChoosed[j][0], intervallsNearUnitCircleChoosed[j][1]))) * random.uniform(radiusCloseToUnitCircle[0], radiusCloseToUnitCircle[1])
            points_ret[j + i * intervallsNearUnitCircleChoosed.shape[0]][1] = math.sin(math.radians(random.uniform(intervallsNearUnitCircleChoosed[j][0], intervallsNearUnitCircleChoosed[j][1]))) * random.uniform(radiusCloseToUnitCircle[0], radiusCloseToUnitCircle[1])
            pass
        pass

    for i in range(rndIntervalls):
        points_ret[-1 -i][0] = math.cos(math.radians(random.uniform(intervallsNearUnitCircleChoosed[i][0], intervallsNearUnitCircleChoosed[i][1]))) * random.uniform(radiusCloseToUnitCircle[0], radiusCloseToUnitCircle[1])
        points_ret[-1 -i][1] = math.sin(math.radians(random.uniform(intervallsNearUnitCircleChoosed[i][0], intervallsNearUnitCircleChoosed[i][1]))) * random.uniform(radiusCloseToUnitCircle[0], radiusCloseToUnitCircle[1])
        pass

    print(checkWhetherPointsLie_Outside_TheUnitCircle(points_ret))

    #return points_ret
    pass

def validData(trainData=1):
    # Unit circle is divided into 12 intervals
    if trainData < 13:
        return withinUnitCircle(trainData)  
        pass
    else:
        pointsOnBorderAndWithinUnitCircle = np.zeros((trainData, 2))
        # Every 13th point should be a border point
        numberOfBorderPoints = int(trainData / 13)
        
        # Points within the 12 circle intervals
        pointsWithinCircleIntervalls = withinUnitCircle(trainData - numberOfBorderPoints)
        for i in range(pointsWithinCircleIntervalls.shape[0]):
            pointsOnBorderAndWithinUnitCircle[i] = pointsWithinCircleIntervalls[i]
            pass

        pointsOnEdge = borderOfUnitCircle(numberOfBorderPoints)
        for i in range(pointsOnEdge.shape[0]):
            pointsOnBorderAndWithinUnitCircle[i + pointsWithinCircleIntervalls.shape[0]] = pointsOnEdge[i]
            pass

        return pointsOnBorderAndWithinUnitCircle
        pass
    pass

def validDataLabeld(trainData=1):
    return labelData(validData(trainData), validDataValue)
    pass

def invalidData(trainData=1):
    #outsideUnitCircle(trainData)

    #return invalidData
    pass

def invalidDataLabeld(trainData=1):
    #return labelData(invalidData(trainData), invalidDataValue)
    pass

def trainData_shuffeld(trainData=1):
    #validDataLabeld
    #invalidDataLabeld

    #return shuffeldTrainData
    pass

# labelvalue is on position -1
# :param2: labelValue: datatype=int
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


print(withinUnitCircle(1000))

