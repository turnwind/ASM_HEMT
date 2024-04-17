import json
import matplotlib.pyplot as plt
import numpy as np

with open("bot_10.json", 'r') as file:
    values = json.load(file)
xx = np.arange(1,11,1)
plt.scatter(xx,values)
bestvalues = []
for i in range(len(values)):
    bestvalues.append(min(values[:i+1]))
plt.plot(bestvalues)
plt.show()
