import numpy as np, numpy
import pandas as pd, pandas
import matplotlib
import collections
import StatMethods as sm
import statistics as stat
import Graphs as grap

data = np.loadtxt("dna_amplification.txt")
data2 = []
data2.append(data)
data = data2
classes = np.genfromtxt("neoplasm_types.txt", delimiter = '\t', dtype='str')

#using pandas:
table = pd.read_csv('dna_amplification.txt', header = None, sep = " ")
rowNames = pd.read_csv("neoplasm_types.txt", header = None, sep ="\t")

#Getting classes
frequencies = collections.Counter(classes[:,1])
print(frequencies)
#Getting the cancer type stats


grap.barPlotPercentage(frequencies.values(),list(frequencies.keys()))


