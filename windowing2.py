import pandas as pd
import numpy as np

def windowing(data: pd.DataFrame, intervalsize: int) -> pd.DataFrame:
    print("windowing...")
    # transform it into numpy array
    array_data = data.values

    # Connect the rows and create new array
    array_data_new = np.empty(intervalsize * array_data.shape[1])
    temp = array_data[i:i + intervalsize].flatten()

    for i in np.arange(0, array_data.shape[0] - intervalsize + 1):
        print(i)
        temp = array_data[i:i + intervalsize].flatten()
        array_data_new = np.vstack([array_data_new, temp])

    # transform back to dataframe
    df_data_new = pd.DataFrame(array_data_new)

    return df_data_new
