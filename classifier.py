from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

'''*********************************************************************************************************************
Random Forest:
Random Forest is an ensemble method that builds multiple decision trees and combines their predictions 
to improve accuracy and reduce overfitting. This is the basic version. More extras in feature engeneering. 
*********************************************************************************************************************'''
def rf_clf(X_train: DataFrame, y_train: DataFrame, n_estimators=100, random_state=42, verbose=2):
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
    rf_clf.fit(X_train, y_train.values.ravel())
    return rf_clf