from pandas import DataFrame
from tqdm import tqdm

def save_csv(data: DataFrame, path: str, chunk_size: int = 200):
    with tqdm(total=len(data), desc="Saving to CSV", unit="row") as pbar:
        for i in range(0, len(data), chunk_size):
            data.iloc[i:i + chunk_size].to_csv(path, mode='a', header=(i == 0), index=False)
            pbar.update(chunk_size)