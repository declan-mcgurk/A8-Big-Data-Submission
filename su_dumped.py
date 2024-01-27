import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pandas as pd

# Plot learning curve
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#80CBC4",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#00897B",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

df1 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/1_0-dft.csv')
df2 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/1_1-dft.csv')
df3 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/2_0-dft.csv')
df4 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/2_1-dft.csv')
df5 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/3_0-dft.csv')
df6 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/3_1-dft.csv')
df7 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/4_0-dft.csv')
df8 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/4_1-dft.csv')
df9 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/5_0-dft.csv')
df10 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/5_1-dft.csv')
df11 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/6_0-dft.csv')
df12 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/6_1-dft.csv')
df13 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/7_0-dft.csv')
df14 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/7_1-dft.csv')
df15 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/0_0-dft.csv')
df16 = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/fourier_transformed_data/0_1-dft.csv')


data1 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14], ignore_index=True)
data2 = pd.concat([df15, df16], ignore_index=True)

X_train = data1.drop(['5000'], axis=1)
y_train = data1['5000']

X_test = data2.drop(['5000'], axis=1)
y_test = data2['5000']

from sklearn.metrics import classification_report
import save
import tensorflow as tf
PATH = './classifiers/ann/ann.txt'
new_clf = save.load_clf(PATH)
print(classification_report(y_test, new_clf.predict(X_test)))


# Generate learning curve
train_sizes, train_scores, valid_scores = learning_curve(new_clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

# Calculate mean and standard deviation of training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid(True)
plt.show()
