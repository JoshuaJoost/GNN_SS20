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

WINDOW_X_INCH = 9.0
WINDOW_Y_INCH = 6.5

def printPerformance(data, summary=False):

    #TODO + errorPerformance: Zusätzliche, eindeutige Markierung für die erreichte Quote auf der linken Skala oder als zusätzliches label
    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)

    ax.plot(data, 's-', markersize=6, color='blue')
    ax.set(xlabel='Epoche(n)', ylabel='Performance (%)', title='Trefferquote je Epoche')
    ax.grid( axis='y', linestyle='--')
    ax.set_xlim(0,)

    return ax if summary is True else plt.show()
    pass


def printErrorPerformance(dataError, summary=False):

    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        fig.suptitle("Statistik über Neuronales Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)

    ax.plot(dataError, 's-', markersize=6, color='darkred')
    ax.set(xlabel='Epoche(n)', ylabel='Fehlerrate (%)', title='Fehlerquote je Epoche')
    ax.grid( axis='y', linestyle='--')
    ax.set_xlim(0,)

    return ax if summary is True else plt.show()
    pass


def printSummary(dataError, dataPerformance, value_range, query):

    fig = plt.figure()
    fig.set_size_inches(WINDOW_X_INCH, WINDOW_Y_INCH)
    
    # TODO kürzerer Code?
    #for add_print in [printErrorPerformance(dataError, summary=True), 
    #            printCircle(value_range, query, summary=True), 
    #            printPerformance(dataPerformance, summary=True)]:


    #ax1 error quote
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1 = printErrorPerformance(dataError, summary=True)
    
    #ax2 circle
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2 = printCircle(value_range, query, summary=True)
    
    #ax3 performance
    ax3 = plt.subplot2grid((2,2),(1,0),colspan=2)
    ax3 = printPerformance(dataPerformance, summary=True)
    
    fig.suptitle('Aktuelle Statistik zum Neuronalen Netzwerk', fontsize=26)
    fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9)
    fig.subplots_adjust(top=0.85)

    plt.show()
    pass

def printCircle(value_range, query, summary=False):

    x_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    y_range = np.arange(-value_range * 2, value_range * 2, 0.05)
    x, y = np.meshgrid(x_range, y_range)
    z = []
    for x_coordinate in x_range:
        z_row = []
        for y_coordinate in y_range:
            z_row.append(query(x_coordinate,y_coordinate))
        z.append(z_row)

    # prepare plot
    if(summary is True):
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        fig.suptitle("Performance-Circle", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)
        fig.suptitle("Aktuelle Statistik zum Neuronalen Netzwerk", fontsize=18, fontweight='bold')
        fig.subplots_adjust(top=0.83)

    ax.set(xlabel='x', ylabel='y', title='Erzielte Verteilung der Daten')
    ax.set_aspect('equal', 'box')
    z = np.array(z)
    z = z.reshape(z.shape[0], z.shape[1])
    p = ax.pcolor(x, y, z)
    plt.colorbar(p)

    return ax if summary is True else plt.show()
    
    pass






