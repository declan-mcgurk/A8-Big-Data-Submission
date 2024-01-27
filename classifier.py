'''*********************************************************************************************************************
Classification algorithms are a type of supervised learning algorithm used for predicting the category or class label
of an instance based on its input features. These algorithms learn from labeled training data and can be used to
classify new, unseen instances into predefined classes. Here are some common classification algorithms:
*********************************************************************************************************************'''

from pandas import DataFrame
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

'''*********************************************************************************************************************
Decision Trees:
Decision Trees recursively split the data based on features to create a tree-like structure. Each leaf node 
represents a class label.
*********************************************************************************************************************'''
def dt_clf(X_train: DataFrame, y_train: DataFrame, random_state=42):
    dt_clf = DecisionTreeClassifier(random_state=random_state)
    dt_clf.fit(X_train, y_train)
    return dt_clf

'''*********************************************************************************************************************
Random Forest:
Random Forest is an ensemble method that builds multiple decision trees and combines their predictions 
to improve accuracy and reduce overfitting. This is the basic version. More extras in feature engeneering. 
*********************************************************************************************************************'''
def rf_clf(X_train: DataFrame, y_train: DataFrame, n_estimators=100, random_state=42, verbose=2):
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
    rf_clf.fit(X_train, y_train.values.ravel())
    return rf_clf

'''*********************************************************************************************************************
K-Nearest Neighbors (KNN):
KNN classifies instances based on the majority class of their k nearest neighbors in the feature space. 
It is a non-parametric and lazy learning algorithm.
*********************************************************************************************************************'''
def knn_clf(X_train: DataFrame, y_train: DataFrame, n_neighbors=3):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust the number of neighbors (k)
    knn_clf.fit(X_train, y_train.values.ravel())
    return knn_clf

def tune(X_train: DataFrame, y_train: DataFrame, estimator, params, scorring="accuracy"):
    cv_method = StratifiedKFold(n_splits=3, random_state=0, shuffle = True)

    GridSearch = GridSearchCV(estimator=estimator,
                                param_grid=params,
                                cv=cv_method,
                                verbose=1,
                                n_jobs=2,
                                scoring=scorring,
                                return_train_score=True
                                )
    
    # Fit model with train data
    GridSearch.fit(X_train, y_train)
    best_estimator = GridSearch.best_estimator_
    print(f"Best estimator for LR model:\n{best_estimator}")

    best_params = GridSearch.best_params_
    print(f"Best parameter values for model:\n{best_params}")
    print(f"Best score for tuned model: {round(GridSearch.best_score_, 3)}")

def adaboost(X_train: DataFrame, y_train: DataFrame, base_estimator=None, n_estimators=50):
    adaboost = AdaBoostClassifier(base_estimator, n_estimators=n_estimators)
    adaboost.fit(X_train, y_train)
    return adaboost