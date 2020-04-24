__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"

# kernel imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# own data imports
import ownFunctions
from ownFunctions import generateNValidTrainDataLabeld, generateNInvalidTrainDataLabeld
import constants
from constants import numberOfValidTrainData, numberOfInvalidTrainData


## Generate Traindata -------------------------------------
validTrainData = generateNValidTrainDataLabeld(numberOfValidTrainData)
invalidTrainData = generateNInvalidTrainDataLabeld(numberOfInvalidTrainData)

### TODO Create, train and validate the neural network and output statistics (view.py)










