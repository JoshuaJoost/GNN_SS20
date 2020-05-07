__authors__ = "Rosario Allegro (1813064), Sedat Cakici (1713179), Joshua Joost (1626034)"
# maintainer = who fixes buggs?
__maintainer = __authors__
__date__ = "2020-04-23"
__version__ = "0.0"
__status__ = "Development"
##--- TODO 
# - Graph showing the learning progress of the neural network
# - Let the neural network draw the unit circle
# - testen

# kernel imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# dummy data
dataError = np.array([0.223, 0.212, 0.201, 0.208, 0.210, 0.203, 0.201, 0.199, 0.198, 0.195, 0.196]).dot(100)
dataPerformance = np.array([0.123, 0.212, 0.302, 0.404, 0.567, 0.654, 0.778, 0.802, 0.823, 0.845, 0.901]).dot(100)

def dummyCircle(dataError, summary=False):

    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
    
    t = np.linspace(0,np.pi*2,100)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)

    return ax
    pass

def dummyPerformance(data, summary=False):
    # prepare plot
    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)

    ax.plot(data, 's-', markersize=6, color='blue')
    ax.set(xlabel='Epoche(n)', ylabel='Performance (%)', title='Trefferquote je Epoche')
    ax.grid( axis='y', linestyle='--')
    ax.set_xlim(0,)

    return ax if summary is True else plt.show()
    pass

def printErrorPerformance(dataError, summary=False):
    # prepare plot
    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)

    ax.plot(dataError, 's-', markersize=6, color='darkred')
    ax.set(xlabel='Epoche(n)', ylabel='Fehlerrate (%)', title='Fehlerquote je Epoche')
    ax.grid( axis='y', linestyle='--')
    ax.set_xlim(0,)

    return ax if summary is True else plt.show()
    pass

#optional
def printSummary():
    fig = plt.figure()
    fig.set_size_inches(9.5, 7.5)
    
    #ax1 error quote
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1 = printErrorPerformance(dataError, summary=True)
    
    #ax2 circle
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2 = dummyCircle(dataError, summary=True)
    
    #ax3 performance
    ax3 = plt.subplot2grid((2,2),(1,0),colspan=2)
    ax3 = dummyPerformance(dataPerformance, summary=True)
    
    fig.suptitle('Aktuelle Statistik zum Neuronalen Netzwerk', fontsize=26)
    fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9)
    fig.subplots_adjust(top=0.85)

    plt.show()
    pass

def printCircle(value_range, query):
    x_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    y_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    x, y = np.meshgrid(x_range, y_range)
    z = []
    for x_coordinate in x_range:
        z_row = []
        for y_coordinate in y_range:
            z_row.append(query(x_coordinate,y_coordinate))
        z.append(z_row)
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    z = np.array(z)
    z = z.reshape(z.shape[0], z.shape[1])
    p = ax.pcolor(x, y, z)
    color_bar = fig.colorbar(p)
    fig.show()
    pass
#-------------
# print error standalone
#printErrorPerformance(dataError)

# print summary
#printSummary()






