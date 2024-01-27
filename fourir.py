from sklearn.metrics import accuracy_score
import pandas as pd
import io
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import save
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PATHFOURIR = "./compdata/dft/6_0-dft.csv"
data_f = pd.read_csv(PATHFOURIR)
df_first_six = data_f.head(6)

column_sums = df_first_six.sum()
df_first_six.loc['sum'] = column_sums

last_row_values = df_first_six.iloc[-1, :].values
df_last_row = pd.DataFrame([last_row_values], columns=df_first_six.columns)


print(df_first_six.tail())
print(df_last_row)


array_data = df_last_row.values
array_data_new = np.delete(array_data, array_data.shape[1] - 1, axis=1)
x = range(0,5000)
plt.scatter(x, array_data_new, s=1, c='#87CEEB')
plt.ylim(np.min(array_data_new)*1.1, np.max(array_data_new)*1.1)
plt.show()
