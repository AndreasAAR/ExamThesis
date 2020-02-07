import numpy as np
import matplotlib.pyplot as plt



def barPlotPercentage(vals,names):
    height = vals
    bars = names

    myList = height
    myInt = sum(height)
    height = [x / myInt for x in myList]

    #range of x axis
    x = (range(len(bars)))
    new_x = [4 * i for i in x]

    y_pos = np.arange(len(x))
    # Create bars
    plt.bar(new_x, height,width=2,align='center')
    # Create names on the x-axis   #Rotate names
    plt.xticks(new_x, bars ,rotation=90,fontsize=5)

    # Show graphic
    plt.figure(figsize=(30, 4))
    plt.show()
