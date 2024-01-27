from pandas import DataFrame
from tqdm import tqdm
from pickle import dump
from pickle import load

def save_csv(data: DataFrame, path: str, chunk_size: int = 200):
    with tqdm(total=len(data), desc="Saving to CSV", unit="row") as pbar:
        for i in range(0, len(data), chunk_size):
            data.iloc[i:i + chunk_size].to_csv(path, mode='a', header=(i == 0), index=False)
            pbar.update(chunk_size)

def save_clf(clf, PATH):
    with open(PATH, "wb") as f:
        dump(clf, f)

def load_clf(PATH):
    with open(PATH, "rb") as f:
        clf = load(f)
    return clf