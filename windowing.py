import pandas as pd
import numpy as np

def windowing(data: pd.DataFrame, intervalsize: int) -> pd.DataFrame:
<<<<<<< HEAD
    print("\tWindowing...")
    reshape = data
    for i in range(1, intervalsize):
        print("\t" + str(i))
        tmp = data
        for j in range(i):
            tmp = tmp.drop(0)
            tmp = tmp.reset_index(drop=True)
        tmp = tmp.reset_index(drop=True)
        reshape = pd.concat([reshape, tmp], axis=1)
    for i in range(intervalsize-1):
        n = reshape.shape[0]
        reshape = reshape.drop(n-1)
        reshape = reshape.reset_index(drop=True)
    reshape = pd.DataFrame.astype(reshape, "float32")
    reshape.columns = range(len(reshape.columns))
    return reshape
=======
    print("windowing...")
    # transform it into numpy array
    array_data = data.values

    # Connect the rows and create new array
    array_data_new = np.empty(intervalsize * array_data.shape[1])

    for i in np.arange(0, array_data.shape[0] - intervalsize + 1):
        print(i)
        temp = array_data[i:i + intervalsize].flatten()
        array_data_new = np.vstack([array_data_new, temp])

    # transform back to dataframe
    df_data_new = pd.DataFrame(array_data_new)

    return df_data_new
>>>>>>> a7e1782bba712191af94b09c46fb4fac89de190d
