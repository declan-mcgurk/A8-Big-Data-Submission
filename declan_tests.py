import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from datapreprocessing import xy_split, tt_split, rnd_sampling
from save import save_clf, load_clf

def classify():
    datas = []
    skip = 1
    for i in range(8):
        if i == skip:
            continue
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

    X = data.iloc[:, :5000]
    y = data.iloc[:, 5000]
    X, y = rnd_sampling(X, y)
    
    print("Training clf...")
    clf = RandomForestClassifier(n_estimators=50, verbose=2, n_jobs = -1)
    clf.fit(X, y)
    print("Trained.")

    print("Saving clf...")
    save_clf(clf, "./classifiers/rf/dft_test-" + str(skip) + ".txt")
    print("Clf saved.")

    # clf = load_clf("./classifiers/rf/trained_1-2-raw.txt")

    # print("Loading new data...")
    # new_data = pd.read_csv("./compdata/bdhsc_2024/stage1_labeled/7_0.csv")
    # print("Loaded.")

    # X_new = new_data.iloc[:, :5000]
    # y_new = new_data.iloc[:, 5000]
    # print(classification_report(y_new, clf.predict(X_new)))

def test():
    data = pd.read_csv("./compdata/dft/1_0-dft.csv")
    X = data.iloc[:, :5000]
    y = data.iloc[:, 5000]
    X, y = rnd_sampling(X, y)
    X_train, X_test, y_train, y_test = tt_split(X, y)

    clf = RandomForestClassifier(n_estimators=20, verbose=2)
    clf.fit(X_train, y_train)
    PATH = "./classifiers/rf/test.txt"
    save_clf(clf, PATH)

    new_clf = load_clf(PATH)
    print(classification_report(y_test, new_clf.predict(X_test)))

if __name__ == "__main__":
    classify()