from numpy import nonzero
from numpy import fft
from pandas import DataFrame, concat, read_csv
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to apply FFT to each row
def apply_fft(row):
    return fft.fft(row)

def fft_df():
    for i in range(8):
            for j in range(2):
                print(str(i) + "_" + str(j) + ":")
                PATH = "./compdata/bdhsc_2024/stage1_labeled/" + str(i) + "_" + str(j) + ".csv"
                df = read_csv(PATH)

                # drop the classification column
                df = df.drop(df.columns[len(df.columns)-1], axis=1)

                # Apply the FFT function to each row with a progress bar
                fft_result = DataFrame()

                # Use tqdm to create a progress bar
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
                    fft_result = concat([fft_result, DataFrame(apply_fft(row)).T], ignore_index=True)

                # The resulting DataFrame will contain the FFT coefficients for each row
                # You may want to take the absolute value or other processing based on your specific needs
                    
                fft_result.to_csv("./compdata/" + str(i) + "_" + str(j) + ".csv")


def pca(X: DataFrame, Y: DataFrame) -> DataFrame:

    pca = PCA()
    pca.fit(X, Y)

    plt.plot(X.keys(), pca.explained_variance_ratio_)
    plt.xticks(rotation=30, ha='right')
    plt.xlabel("X")
    plt.ylabel("Explained variance")
    plt.show()

    plt.plot(cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of features")
    plt.ylabel("Explained variance")
    plt.show()

    return pca