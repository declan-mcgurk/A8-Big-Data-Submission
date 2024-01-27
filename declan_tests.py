import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from featureengineering import rnd_sampling
from datapreprocessing import xy_split, tt_split

def classify():
    print("Reading data 1...")
    data_1 = pd.read_csv("./compdata/dft/1_0-dft.csv")
    print(data_1.shape)
    print("Reading data 2...")
    data_2 = pd.read_csv("./compdata/dft/2_0-dft.csv")
    print(data_2.shape)
    print("Reading data 3...")
    data_3 = pd.read_csv("./compdata/dft/3_0-dft.csv")
    print(data_3.shape)
    print("Reading data 4...")
    data_4 = pd.read_csv("./compdata/dft/4_0-dft.csv")
    print(data_4.shape)
    print("All data read.")
    data = pd.concat([data_1, data_2, data_3, data_4], axis=0)
    print(data.shape)

    X = data.iloc[:, :5000]
    y = data.iloc[:, 5000]

    X, y = rnd_sampling(X, y)
    print(X)
    print(y)
    
    X_train, X_test, y_train, y_test = tt_split(X, y)
    
    clf = RandomForestClassifier(verbose=2)
    print("Training clf...")
    clf.fit(X_train, y_train)
    print("Trained.")
    print(classification_report(y_test, clf.predict(X_test)))

    print("Loading new data...")
    new_data = pd.read_csv("./compdata/dft/7_0-dft.csv")
    print("Loaded.")
    print(data)

    X_new = new_data.iloc[:, :5000]
    y_new = new_data.iloc[:, 5000]
    print(X_new)
    print(y_new)
    print(classification_report(y_new, clf.predict(X_new)))

if __name__ == "__main__":
    classify()