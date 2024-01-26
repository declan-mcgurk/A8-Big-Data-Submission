import pandas as pd
import numpy as np
from tqdm import tqdm

from datapreprocessing import find_missing

# Function to apply FFT to each row
def apply_fft(row):
    return np.fft.fft(row)

def main():
    # import the data
    PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
    df = pd.read_csv(PATH)

    # # Apply the FFT function to each row with a progress bar
    # fft_result = pd.DataFrame()

    # # Use tqdm to create a progress bar
    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
    #     fft_result = pd.concat([fft_result, pd.DataFrame(apply_fft(row)).T], ignore_index=True)

    # # The resulting DataFrame will contain the FFT coefficients for each row
    # # You may want to take the absolute value or other processing based on your specific needs
        
    # fft_result.to_csv("./compdata/0_0-dft.csv")

if __name__ == "__main__":
    main()