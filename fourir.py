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

PATHFOURIR = "./compdata/dft/4_0-dft.csv"
data_f = pd.read_csv(PATHFOURIR)
data_f_six = data_f.head(6)
print(data_f_six.head())
data_f_six['sum'] = data_f_six.sum(axis=1)

print(data_f_six.shape)
print(data_f_six['sum'].head())

x = range(0,5001)
plt.scatter(x, data_f_six['sum'], s=1, c='#87CEEB')
plt.vlines(range(0, 5000), -10, 10, colors = '#6495ED')
plt.ylim(np.min(data_f_six)*1.1, np.max(data_f_six)*1.1)
plt.show()
