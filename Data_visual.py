import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open("Datafortranin//paras2.json") as f:
    dataY = json.load(f)
dataY = np.array(dataY)
dataY = dataY[1:,:]

DC_paras = ["voff", "nfactor", "u0", "ua", "Igsdio", "Njgs", "Igddio", "Njgd", "Rshg", "Eta0", "Vdscale", "Cdscd",
            "Rsc", "Rdc", "UTE", "RTH0", "LAMBDA", "Vsat", "Tbar"]
df = pd.DataFrame(dataY, columns=DC_paras)
# Histogram visualization for each feature
for column in df.columns:
    plt.figure(figsize=(10,5))
    sns.histplot(df[column], kde=False, bins=30)
    plt.title(f'Distribution of {column}')
    plt.show()