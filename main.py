import pandas as pd
import numpy as np
from tqdm import tqdm

from datapreprocessing import find_missing
from featureengineering import fft_df
from windowing import windowing
from save import save_csv

def main():
<<<<<<< HEAD
    for i in range(8):
        for j in range(2):
            NUM = str(i) + "_" + str(j)
            print(NUM + ":")
            print("\tLoading data...")
            data = pd.read_csv("./compdata/bdhsc_2024/stage1_labeled/" + NUM + ".csv")
            print("\tData loaded.")
            X = data.drop("5000", axis=1)
            data_fft = fft_df(X)
            data_fft = pd.concat([data_fft, data.iloc[:,5000]], axis=1)
            save_csv(data_fft, "./compdata/dft/" + NUM + "-dft.csv")
            data_fft_win = windowing(data_fft, 5)
            save_csv(data_fft_win, "./compdata/win/" + NUM + "-win.csv")
=======
    PATH = "./compdata/0_0.csv"
    data = pd.read_csv(PATH)

    new = windowing(data, 5)
    print(new)
    print(new.shape)
>>>>>>> a7e1782bba712191af94b09c46fb4fac89de190d

if __name__ == "__main__":
    main()