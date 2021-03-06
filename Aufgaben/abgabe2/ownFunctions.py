__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "1.0"
__status__ = "ReadyForUse"

# kernel imports
import numpy as np
import random
import math

# own data imports
import constants
from constants import xMin, xMax, inputNeurons, invalidTrainDataMaxPoint, invalidTrainDataMinPoint, invalidTrainDataExklusivPointDistance, validDataValue, invalidDataValue
import ownTests

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

# Tested with checkForInvalidData_Outside_TheAreaNearTheUnitCircle, Method OK
def outsideUnitCircle(points=1):
    if points == 0:
        return np.array((0,2))
        pass

    pointsOutsideUnitCircle = np.zeros((points, 2))
    intervallsDegrees = np.array([
        [0, 90 - 0.001],
        [90, 180 - 0.001],
        [180, 270 - 0.001],
        [270, 360 - 0.001]
    ])

    largestDistancesInIntervalls = np.array([
        # [0] centre [1] 0 <= alpha <= 45 Degree; [2] 45 <= alpha <= 90 Degree
        [45, lambda alpha: constants.xMax / math.cos(alpha), lambda alpha: constants.xMax / math.sin(alpha)],
        # [0] centre [1] 90 <= alpha <= 135 Degree; [2] 135 <= alpha <= 180 Degree
        [135, lambda alpha: constants.xMax / math.sin(alpha), lambda alpha: -(constants.xMax / math.cos(alpha))],
        # [0] centre [1] 180 <= alpha <= 225 Degree; [2] 225 <= alpha <= 270
        [225, lambda alpha: -(constants.xMax / math.cos(alpha)), lambda alpha: -(constants.xMax / math.sin(alpha))],
        # [0] centre [1] 270 <= alpha <= 315 Degree; [2] 315 <= alpha <= 360
        [315, lambda alpha: -(constants.xMax / math.sin(alpha)), lambda alpha: constants.xMax / math.cos(alpha)]
    ])

    intervallsChoosed = None
    if points < intervallsDegrees.shape[0]:
        intervallsChoosed = np.zeros([points, intervallsDegrees.shape[1]])

        # choose random intervalls
        for i in range(intervallsChoosed.shape[0]):
            choosedIntervall = random.randint(0, intervallsDegrees.shape[0] - 1)
            intervallsChoosed[i] = intervallsDegrees[choosedIntervall]
            intervallsDegrees = np.delete(intervallsDegrees, choosedIntervall, 0)
            pass
        pass
    else:
        intervallsChoosed = intervallsDegrees
        pass

    rndIntervalls = points % intervallsChoosed.shape[0]

    for i in range(int(points / intervallsChoosed.shape[0])):
        for j in range(intervallsChoosed.shape[0]):
            degree = random.uniform(intervallsChoosed[j][0], intervallsChoosed[j][1])
            alpha = math.radians(degree)
            radius = 0
            if degree <= largestDistancesInIntervalls[j][0]:
                radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[1], largestDistancesInIntervalls[j][1](alpha))
                pass
            else:
                radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[1], largestDistancesInIntervalls[j][2](alpha))
                pass
            pointsOutsideUnitCircle[j + i * intervallsChoosed.shape[0]][0] = math.cos(alpha) * radius
            pointsOutsideUnitCircle[j + i * intervallsChoosed.shape[0]][1] = math.sin(alpha) * radius
            pass
        pass

    for i in range(rndIntervalls):
        degree = random.uniform(intervallsChoosed[j][0], intervallsChoosed[j][1])
        alpha = math.radians(degree)
        radius = 0
        if degree <= largestDistancesInIntervalls[j][0]:
            radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[1], largestDistancesInIntervalls[j][1](alpha))
            pass
        else:
            radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[1], largestDistancesInIntervalls[j][2](alpha))
            pass
        pointsOutsideUnitCircle[-1 -i][0] = math.cos(alpha) * radius
        pointsOutsideUnitCircle[-1 -i][1] = math.sin(alpha) * radius
        pass

    return pointsOutsideUnitCircle
    pass

# Tested with checkWheterPointsLie_Outside_butCloseUnitCircleBorder, Method OK
def points_Outside_CloseToUniCircleBorder(points=1):
    if points == 0:
        return np.array((0,2))
        pass

    points_ret = np.zeros((points, 2))

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
            alpha = random.uniform(intervallsNearUnitCircleChoosed[j][0], intervallsNearUnitCircleChoosed[j][1])
            radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[0], constants.radiusIntervalCloseToUnicircleBorder[1])
            points_ret[j + i * intervallsNearUnitCircleChoosed.shape[0]][0] = math.cos(math.radians(alpha)) * radius
            points_ret[j + i * intervallsNearUnitCircleChoosed.shape[0]][1] = math.sin(math.radians(alpha)) * radius
            pass
        pass

    for i in range(rndIntervalls):
        alpha = random.uniform(intervallsNearUnitCircleChoosed[i][0], intervallsNearUnitCircleChoosed[i][1])
        radius = random.uniform(constants.radiusIntervalCloseToUnicircleBorder[0], constants.radiusIntervalCloseToUnicircleBorder[1])
        points_ret[-1 -i][0] = math.cos(math.radians(alpha)) * radius
        points_ret[-1 -i][1] = math.sin(math.radians(alpha)) * radius
        pass

    return points_ret
    pass

# Tested with checkWhetherPointsLie_Inside_TheUnitCircle, Error at the border of the values of the unit circle in the range of 1**-16, Method OK
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

# Tested with checkWhetherPointsLie_Outside_TheUnitCircle, Method OK
def invalidData(trainData=1):
    invalidData = np.zeros((trainData, 2))
    numberOfPointsBeyondUnitCircle = int(trainData / 2.5)
    numberOfPointsNearUnitCircle = trainData - numberOfPointsBeyondUnitCircle

    pointsBeyondUnitCircle = outsideUnitCircle(numberOfPointsBeyondUnitCircle)
    for i in range(numberOfPointsBeyondUnitCircle):
        invalidData[i][0] = pointsBeyondUnitCircle[i][0]
        invalidData[i][1] = pointsBeyondUnitCircle[i][1]
        pass

    pointsNearUnitCircle = points_Outside_CloseToUniCircleBorder(numberOfPointsNearUnitCircle)
    for i in range(numberOfPointsNearUnitCircle):
        invalidData[-1 - i][0] = pointsNearUnitCircle[i][0]
        invalidData[-1 - i][1] = pointsNearUnitCircle[i][1]
        pass

    return invalidData
    pass

def invalidDataLabeld(trainData=1):
    return labelData(invalidData(trainData), invalidDataValue)
    pass

def trainData_shuffeld(trainData=1):
    trainDataShuffeld = np.zeros((trainData, 2))
    numberOfValidData = int(trainData / 2)
    numberOfInvalidData = int(trainData / 2)

    if trainData == 1 or trainData % 2 == 1:
        rnd = random.randint(0, 1)
        if rnd == 0:
            numberOfValidData += 1
            pass
        else:
            numberOfInvalidData += 1
            pass
        pass

    _validData = validData(numberOfValidData)
    for i in range(numberOfValidData):
        trainDataShuffeld[i][0] = _validData[i][0]
        trainDataShuffeld[i][1] = _validData[i][1]
        pass

    _invalidData = invalidData(numberOfInvalidData)
    for i in range(numberOfInvalidData):
        trainDataShuffeld[-1 - i][0] = _invalidData[i][0]
        trainDataShuffeld[-1 - i][1] = _invalidData[i][1]
        pass

    np.random.shuffle(trainDataShuffeld)
    return trainDataShuffeld
    pass

def trainDataLabeld_shuffeld(trainData=1):
    trainDataLabeldShuffeld = np.zeros((trainData, 3))
    numberOfValidDataLabeld = int(trainData / 2)
    numberOfInvalidDataLabeld = int(trainData / 2)

    if trainData == 1 or trainData % 2 == 1:
        rnd = random.randint(0, 1)
        if rnd == 0:
            numberOfValidDataLabeld += 1
            pass
        else:
            numberOfInvalidDataLabeld += 1
            pass
        pass

    _validDataLabeld = validDataLabeld(numberOfValidDataLabeld)
    for i in range(numberOfValidDataLabeld):
        trainDataLabeldShuffeld[i][0] = _validDataLabeld[i][0]
        trainDataLabeldShuffeld[i][1] = _validDataLabeld[i][1]
        trainDataLabeldShuffeld[i][2] = _validDataLabeld[i][2]
        pass

    _invalidDataLabeld = invalidDataLabeld(numberOfInvalidDataLabeld)
    for i in range(numberOfInvalidDataLabeld):
        trainDataLabeldShuffeld[-1 - i][0] = _invalidDataLabeld[i][0]
        trainDataLabeldShuffeld[-1 - i][1] = _invalidDataLabeld[i][1]
        trainDataLabeldShuffeld[-1 - i][2] = _invalidDataLabeld[i][2]
        pass

    np.random.shuffle(trainDataLabeldShuffeld)
    return trainDataLabeldShuffeld
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

def getRandomTrainData(numberOfTrainData = 1):
    trainData = np.zeros((numberOfTrainData, 3))

    for i in range(trainData.shape[0]):
        rndX = random.uniform(-2, 2)
        rndY = random.uniform(-2, 2)
        targetValue = None

        if rndX**2 + rndY**2 <= 1:
            targetValue = constants.validDataValue
            pass 
        else:
            targetValue = constants.invalidDataValue
            pass

        trainData[i] = np.array([rndX, rndY, targetValue])
        pass

    return trainData
    pass


#print(validDataLabeld(1))

