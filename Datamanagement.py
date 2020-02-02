import numpy

data = numpy.loadtxt("dna_amplification.txt")
data2 = []
data2.append(data)
data = data2
classes = numpy.genfromtxt("neoplasm_types.txt", delimiter = '\t', dtype='str')

print(data[:10])
print(classes[:10])

data.append(classes)
print(data[:10])