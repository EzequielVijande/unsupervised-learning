from typing import Sequence, Tuple
import numpy as np
import pandas as pd

def zscore(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Normal Standardization (Z-Score): (X - mean) / std"""
    X = df.loc[:, list(cols)].to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd, mu.ravel(), sd.ravel()
