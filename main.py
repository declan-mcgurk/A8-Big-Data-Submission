import pandas as pd
import numpy as np
from tqdm import tqdm

from datapreprocessing import find_missing
from featureengineering import fft_df
from windowing import windowing

def main():
    PATH = "./compdata/dft/0_0-dft.csv"
    data = pd.read_csv(PATH)

    new = windowing(data, 5)
    print(new)
    print(new.shape)

if __name__ == "__main__":
    main()