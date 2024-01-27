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

from datapreprocessing import xy_split, tt_split, rnd_sampling
from save import save_clf, load_clf


def plot_Imbalanced(data: pd.DataFrame, rat: int):
    class_distribution = data.iloc[:, -1].value_counts()
    #print("imbalance: ")
    #print(class_distribution)
    class_distribution.plot(kind='bar', rot=0, color=['#87CEEB', '#6495ED', '#4169E1'])
    plt.title('Label Distribution')
    plt.xlabel('Lable')
    plt.ylabel('Count')
    custom_labels = ["Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"]
    plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=0)
    plt.show()
    plt.savefig('Imbalanced_data' + str(rat))


def plot_Data(data: pd.DataFrame, start: int, length: int, rat: int):
    array_data = data.values
    array_data_new = np.delete(array_data, array_data.shape[1] - 1, axis=1)
    array_flatten = array_data_new[start:start+length].flatten()

    x = range(start,length*5000)
    plt.scatter(x, array_flatten, s=1, c='#87CEEB')
    plt.vlines(range(start, length*5000, 5000,), -10, 10, colors = '#6495ED')
    plt.ylim(np.min(array_flatten), np.max(array_flatten))
    plt.show()
    plt.savefig('data_plot' + str(length) + "rat:_" + str(rat))


def plot_classification_report(y_true, y_pred, rat: int):
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Plot heatmap for precision, recall, and f1-score
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_report[['precision', 'recall', 'f1-score']], annot=True, cmap=sns.light_palette("skyblue", as_cmap=True), fmt=".3f", linewidths=.5, xticklabels=('Precision', 'Recall', 'F1-score'), yticklabels=("Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Classification Report')
    plt.show()
    plt.savefig("CM" + str(rat))


# Function to generate and plot a confusion matrix
def plot_confusion_matrix_en(y_true, y_pred, rat: int):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot heatmap for the confusion matrix
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette("skyblue", as_cmap=True), linewidths=.5, annot_kws={"size": 16}, xticklabels=("Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"), yticklabels=("Paradoxical Sleep", "Slow-Wave Sleep", "Wakefulness"))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig("CM" + str(rat))






PATHDFT = "C:/Users/hanna/OneDrive/Dokumente/#Florida/#Spring Semester/A8-Big-Data-Submission/compdata/dft/7_0-dft.csv"
data = pd.read_csv(PATHDFT)
X = data.drop('5000', axis=1)
y = data['5000']
X = data.drop('5000', axis=1)
y = data['5000']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = save.load_clf("./classifiers/rf/dft_test-1.txt")
y_pred = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("accuracy")

# test reports rf, rat 7
plot_classification_report(y_pred, y_test, 7)
plot_confusion_matrix_en(y_pred, y_test, 7)




#test imbalanced original data 0 and all

PATHDFT = "./compdata/bdhsc_2024/stage1_labeled/0_0.csv"
data = pd.read_csv(PATHDFT)

plot_Imbalanced(data, 0)
plot_Data(data, 50, 10, 0)


datas = []
for i in range(8):
    #if i == skip:
     #   continue
    print("Reading data " + str(i) + "...")
    tmp0 = pd.read_csv("./compdata/dft/" + str(i) + "_0-dft.csv")
    tmp1 = pd.read_csv("./compdata/dft/" + str(i) + "_1-dft.csv")
    if i == 0:
        tmp0 = tmp0.iloc[:, 1:5002]
        tmp1 = tmp1.iloc[:, 1:5002]
    datas.append(tmp0.astype("float32"))
    datas.append(tmp1.astype("float32"))

print("Shapes:")
for df in datas:
    print(df.shape)

data = pd.concat(datas, axis=0).astype("float32")

print("final shape:")
print(data.shape)

plot_Imbalanced(data, 10)