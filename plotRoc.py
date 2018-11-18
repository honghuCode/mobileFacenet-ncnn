import matplotlib.pyplot as plt
import numpy as np

TP = []
FP = []
with open("roc.txt") as f:
    lines = f.readlines()
for line in lines:
	line = line.strip()
	arr = line.split(" ")
	TP.append(float(arr[1]))
	FP.append(float(arr[2]))
	print arr

plt.plot(FP, TP)
plt.savefig('roc.jpg')
plt.show()