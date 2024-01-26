import pandas as pd
import matplotlib.pyplot as plt


data_final = pd.DataFrame(columns=range(5001))

for i in range(0,7):
    for j in range(0,1):
        PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
        data = pd.read_csv(PATH)


print(data_final.shape)


array_data_final = data_final.values
array_flatten_final = array_data_final[:5].flatten()
plt.plot(array_flatten[25000])
