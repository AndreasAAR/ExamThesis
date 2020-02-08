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
    new_x = [6 * i for i in x]

    y_pos = np.arange(len(x))
    # Create bars
    plt.bar(new_x, height,width=2,align='center')
    # Create names on the x-axis   #Rotate names
    plt.xticks(new_x, bars ,rotation=90,fontsize=6.5)
    plt.yticks(rotation = 90)

    # Show graphic
    #plt.figure(figsize=(30, 4))
    plt.tight_layout()
    fig1 = plt.gcf()
    fig1.savefig('Class Division.png', bbox_extra_artists=(plt.xticks()), bbox='tight')
    plt.show()
