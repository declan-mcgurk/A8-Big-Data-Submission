import pandas as pd
import matplotlib.pyplot as plt

i=0
j=0

PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
data = pd.read_csv(PATH)

array_data = data.values
array_flatten = array_data[:5].flatten()
print(plt.plot(array_flatten[25000]))
