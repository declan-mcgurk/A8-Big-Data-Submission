import pandas as pd
import numpy as np
import datapreprocessing


for i in range(0,8):
    for j in range(0,2):
        PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
        data = pd.read_csv(PATH)
        print(i,j)
        X = data.drop('5000', axis=1)
        y = data['5000']
        (X_b, y_b) = datapreprocessing.rnd_sampling(X, y, sample_type="ros")

