import pandas as pd
import matplotlib.pyplot as plt

i=0
j=0

PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
data = pd.read_csv(PATH)
data = data.drop("5000", axis=1)

print(data.head())

array_data = data.values
array_flatten = array_data[:100].flatten()[:]
plt.plot(array_flatten)
plt.show()