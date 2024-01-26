import pandas as pd
import matplotlib.pyplot as plt

count_lable_0 = 0
count_lable_1 = 0
count_lable_2 = 0

for i in range(0,8):
    for j in range(0,2):
        PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
        data = pd.read_csv(PATH)

        class_distribution = data.iloc[:, -1].value_counts()
        print("imbalance:", i, j)
        print(class_distribution)

        # Balkendiagramm für die Klassenhäufigkeit
        class_distribution.plot(kind='bar', rot=0)
        plt.title('Imbalanced Dataset')
        plt.xlabel('Lable')
        plt.ylabel('Count')
        plt.show()
        plt.savefig('mein_plot.png')


        count_lable_0 = count_lable_0 + len(data[data.iloc[:,-1] == 0])
        count_lable_1 = count_lable_1 + len(data[data.iloc[:,-1] == 1])
        count_lable_2 = count_lable_2 + len(data[data.iloc[:,-1] == 2])


print("count lable 0: ", count_lable_0)
print("count lable 1: ", count_lable_1)
print("count lable 2: ", count_lable_2)
'''
data_label_0 = data[data.iloc[:,-1] == 0]
data_label_1 = data[data.iloc[:,-1] == 1]
data_label_2 = data[data.iloc[:,-1] == 2]
print(data_label_0.head())
'''


#
