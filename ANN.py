import pandas as pd
from imblearn.over_sampling import RandomOverSampler

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
'''

CODE FOR WINDOWED DATA
selected_columns = ['5000', '10001', '15002', '20003', '25004']
data['lable'] = round(data[selected_columns].mean(axis=1))
X = data.drop(['5000', '10001', '15002', '20003', '25004', 'lable'], axis=1)
y = data['lable']
'''

X_train = data1.drop(['5000'], axis=1)
y_train = data1['5000']

X_test = data2.drop(['5000'], axis=1)
y_test = data2['5000']

'''
ros = RandomOverSampler(random_state=42)
X_train,y_train = ros.fit_resample(X_train, y_train)
'''
'''
print(X_experiment.shape)

X_experiment.to_csv('windowed_data_X_0_0.csv', index=False)
y_experiment.to_csv('windowed_data_y_0_0.csv', index=False)

X_experiment = pd.read_csv('windowed_data_X_0_0.csv')
y_experiment = pd.read_csv('windowed_data_y_0_0.csv')
concatenated_data = pd.concat([X_experiment, y_experiment], axis=0, ignore_index=True)
print(concatenated_data.head())

X = X_experiment
y = y_experiment

'''
'''
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''
# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


print("We Enter into the World of training!")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Define the model
model = Sequential()
model.add(Dense(units=512, input_dim=5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))  # Output layer with softmax activation for three classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_onehot, epochs=50, batch_size=64, validation_split=0.2)

import save
PATH = "./classifiers/ann/ann.txt"
save.save_clf(model, PATH)


# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()