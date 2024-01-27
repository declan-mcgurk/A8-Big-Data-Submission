import pandas as pd
import numpy as np
from tqdm import tqdm

from datapreprocessing import find_missing
from featureengineering import fft_df
from windowing import windowing
from save import save_csv

def main():
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

if __name__ == "__main__":
    main()