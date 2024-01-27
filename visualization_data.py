import pandas as pd
import matplotlib.pyplot as plt


<<<<<<< HEAD
PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
data = pd.read_csv(PATH)
data = data.drop("5000", axis=1)

print(data.head())

array_data = data.values
array_flatten = array_data[:100].flatten()[:]
plt.plot(array_flatten)
plt.show()
=======
data_final = pd.DataFrame(columns=range(5001))

for i in range(0,7):
    for j in range(0,1):
        PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
        data = pd.read_csv(PATH)


print(data_final.shape)


array_data_final = data_final.values
array_flatten_final = array_data_final[:5].flatten()
plt.plot(array_flatten[25000])
>>>>>>> a7e1782bba712191af94b09c46fb4fac89de190d
