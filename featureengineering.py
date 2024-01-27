from numpy import nonzero
from numpy import fft
from numpy import abs
from pandas import DataFrame, concat, read_csv
from tqdm import tqdm

# Function to apply FFT to each row
def apply_fft(row):
    return fft.fft(row)

def fft_df(df: DataFrame) -> DataFrame:
    print("\tRunning fft...")
    fft_result = DataFrame(abs(fft.fft(df, axis=1)))
    print("\tfft finished.")
    return fft_result