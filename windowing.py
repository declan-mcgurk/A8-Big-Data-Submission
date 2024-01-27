import pandas as pd
import numpy as np

def windowing(data: pd.DataFrame, intervalsize: int) -> pd.DataFrame:
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