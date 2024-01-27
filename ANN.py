
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('C:/Users/sudak/A8-Big-Data-Submission/bdhsc_2024\stage1_labeled/0_0.csv')

'''CODE FOR WINDOWED DATA
selected_columns = ['5000', '10001', '15002', '20003', '25004']
data['lable'] = round(data[selected_columns].mean(axis=1))
X = data.drop(['5000', '10001', '15002', '20003', '25004', 'lable'], axis=1)
y = data['lable']
'''

X = data.drop(['5000'], axis=1)
y = data['5000']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)


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

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''

# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

'''
bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

binned_data = X_experiment.apply(lambda col: pd.cut(col, bins=bin_edges, labels=False))

X_experiment = X_experiment.drop('lable')
print(X_experiment.head())
'''
'''
from keras.models import Sequential
from keras.layers import Dense, Activation

classifier = Sequential()
# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
classifier.add(Dense(units=5, input_dim=25001, kernel_initializer='uniform', activation='relu'))

#Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Defining the Output layer
# sigmoid means sigmoid activation function
# for Multiclass classification the activation ='softmax'
# And output_dim will be equal to the number of factor levels
classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

# Optimizer== the algorithm of SGG to keep updating weights
# loss== the loss function to measure the accuracy
# metrics== the way we will compare the accuracy after each step of SGD
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the Neural Network on the training data
ANN_Model=classifier.fit(X_train,y_train, batch_size=10 , epochs=10, verbose=1)
'''






'''
def make_classification_ann(Optimizer_Trial):
    from keras.models import Sequential
    from keras.layers import Dense

    # Creating the classifier ANN model
    classifier = Sequential()
    classifier.add(Dense(units=5, input_dim=25001, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))
    classifier.compile(Neurons_trial=5,optimizer=Optimizer_Trial, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


########################################

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

Parameter_Trials = {'batch_size': [10, 20, 30],
                    'epochs': [10, 20],
                    'Optimizer_Trial': ['adam', 'rmsprop'],
                    'Neurons_Trial': [5, 10]
                    }

# Creating the classifier ANN
classifierModel = KerasClassifier(make_classification_ann, verbose=0)

########################################

# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search = GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring='f1', cv=5)

########################################

# Measuring how much time it took to find the best params
import time

StartTime = time.time()

# Running Grid Search for different paramenters
grid_search.fit(X_train, y_train, verbose=1)

EndTime = time.time()
print("############### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes #############')

########################################
'''
'''
# printing the best parameters
print('\n#### Best hyperparamters ####')
grid_search.best_params_
'''






'''
from keras.models import Sequential
from keras.layers import Dense, Activation

# Defining a function for finding best hyperparameters
def FunctionFindBestParams(X_train, y_train):
    # Defining the list of hyper parameters to try
    TrialNumber = 0
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [5, 10, 50, 100]

    import pandas as pd
    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1

            # Creating the classifier ANN model
            classifier = Sequential()
            classifier.add(Dense(units=10, input_dim=25001, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            ann_model = classifier.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial,
                                               verbose=0)
            # Fetching the accuracy of the training
            Accuracy = ann_model.history['accuracy'][-1]

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', Accuracy)

            SearchResultsData = SearchResultsData.append(pd.DataFrame(data=[[TrialNumber,'batch_size'+str(batch_size_trial)+'-'+'epoch'+str(epochs_trial), Accuracy]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)

###############################################

# Calling the function
ResultsData=FunctionFindBestParams(X_train, y_train)
'''
'''
X_experiment = X_experiment.drop(['lable'], axis=1)
'''


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Define the model
model = Sequential()
model.add(Dense(units=128, input_dim=5000, activation='relu'))
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

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))