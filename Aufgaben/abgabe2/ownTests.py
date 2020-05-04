__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-05-04"
__version__ = "0.0"
__status__ = "Development"

# kernel imports
import numpy as np


def isUniCirclePoint(xCoord, yCoord):
    if xCoord**2 + yCoord**2 <= 1:
        return True
        pass
    else:
        return False
    pass

def checkWhetherPointsLie_Outside_TheUnitCircle(pointsArray):
    output = ""

    for i in range(pointsArray.shape[0]):
        output += "(" + str(pointsArray[i][0]) + "," + str(pointsArray[i][1])
        if isUniCirclePoint(pointsArray[i][0], pointsArray[i][1]):
            output += ") NOT OK"
            pass
        else:
            output += ") OK"
            pass

        if i + 1 < pointsArray.shape[0]:
            output += "\n"
            pass
        pass

    return output
    pass

def checkWhetherPointsLie_Inside_TheUnitCircle(pointsArray):
    output = ""

    for i in range(pointsArray.shape[0]):
        output += "(" + str(pointsArray[i][0]) + "," + str(pointsArray[i][1])
        if isUniCirclePoint(pointsArray[i][0], pointsArray[i][1]):
            output += ") OK"
            pass
        else:
            output += ") NOT OK"
            pass

        if i + 1 < pointsArray.shape[0]:
            output += "\n"
            pass
        pass

    return output
    pass