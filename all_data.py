import pandas as pd
import os
import matplotlib.pyplot as plt


'''directory_path = './compdata/bdhsc_2024/stage1_labeled'
dataframes = []

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)

        # CSV-Datei read in and add to the DataFrame
        df = pd.read_csv(file_path)
        dataframes.append(df)


big_dataframe = pd.concat(dataframes, ignore_index=True)
print(big_dataframe)
'''

PATHDFT = "./compdata/bdhsc_2024/stage1_labeled/0_0.csv"
big_dataframe = pd.read_csv(PATHDFT)

class_distribution = big_dataframe.iloc[:, -1].value_counts()
print("imbalance: ")
print(class_distribution)

# plot Imbalance
class_distribution.plot(kind='bar', rot=0, color=['#87CEEB', '#6495ED', '#4169E1'])
plt.title('Imbalanced Dataset')
plt.xlabel('Lable')
plt.ylabel('Lable distribution')
plt.legend(["Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"], shadow=True, fontsize=12, loc='upper left')
plt.show()
plt.savefig('mein_plot.png')


#plot Data
array_data_final = data_final.values
array_flatten_final = array_data_final[:5].flatten()
plt.plot(array_flatten[25000])






'''

#create 10% dataframe

# Calculate the number of rows to save (10% of the original DataFrame)
percentage_to_save = 0.1
num_rows_to_save = int(len(big_dataframe) * percentage_to_save)

# Randomly sample 10% of the data
sampled_data = big_dataframe.sample(n=num_rows_to_save, random_state=42)  # Adjust the random_state if needed

# Save the sampled data to a new DataFrame
extra_dataframe = pd.DataFrame(sampled_data)

# Optionally, you can reset the index of the new DataFrame
extra_dataframe.reset_index(drop=True, inplace=True)

# Display the new DataFrame
print(extra_dataframe)'''

#save dataframe
#extra_dataframe.to_csv('./A8-Big-Data-Submission', index=False)
#big_dataframe.to_csv('./A8-Big-Data-Submission', index=False)

