import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import io
import save
import model_Hannah


PATHDFT = "./compdata/bdhsc_2024/stage1_labeled/0_0.csv"
data = pd.read_csv(PATHDFT)


'''# plot Imbalance
class_distribution = data.iloc[:, -1].value_counts()
print("imbalance: ")
print(class_distribution)
class_distribution.plot(kind='bar', rot=0, color=['#87CEEB', '#6495ED', '#4169E1'])
plt.title('Label Distribution')
plt.xlabel('Lable')
plt.ylabel('Count')
custom_labels = ["Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"]
plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=0)
plt.show()
plt.savefig('Imbalanced_data')'''



'''#plot Data
n=8640
array_data = data.values
array_data_new = np.delete(array_data, array_data.shape[1] - 1, axis=1)
array_flatten = array_data_new[:n].flatten()
print(array_flatten)
x = range(n*5000)
plt.scatter(x, array_flatten, s=1, c='#87CEEB')
plt.vlines(range(0, n*5000, 5000,), -10, 10, colors = '#6495ED')
linestyles = ("solid", "dashed", "dotted")
plt.ylim(np.min(array_flatten), np.max(array_flatten))
plt.show()
plt.savefig('data_plot' + str(n))'''

#accuracy

rf = save.load_clf("./classifiers/rf/dft_test-1.txt")
y_pred = rf.predict(model_Hannah.X_test)


# Accuracy
print("accuracy")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Print classification report
print("Classification Report:")
report = classification_report(y_test, y_pred, target_names=["Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"])
print(report)
df_report = pd.read_csv(io.StringIO(report), sep='\s+')
plt.figure(figsize=(8, 6))
sns.heatmap(df_report.iloc[:-3, 1:].astype(float), annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title('Classification Report Heatmap')
plt.show()
plt.savefig('Classification_Report')

