# Create data with different sizes
from typing import Dict, List
import pandas as pd
import numpy as np


def data_generator(data: pd.DataFrame,
                   include_noise: List[str] = None,
                   sizes: List[int] = None,
                   filters: List[str] = None) -> Dict[str,
                                                      pd.DataFrame]:
    """Generates datasets of different sizes
    sampling from the original data
    Args:
        data (pd.DataFrame): Original data
        include_noise (List[str]): List of variables to add noise.
        Defaults to None.
        sizes (List[int], optional): List with different sizes.
        Defaults to None.
        filters (List[str], optional): Filters to apply to the data.
        Each filter will create a different DataFame.
        Defaults to None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys as the data size
        and values with the DataFrame with the respect size.
    """
    if filters:
        datasets = [data.query(f) for f in filters]
        datasets = {str(len(d)): d for d in datasets}
    elif sizes:
        datasets = {str(s): resample_data(data, s)
                    for s in sizes}

    if include_noise:
        datasets = {k: apply_noise(d, include_noise)
                    for k, d in datasets.items()}

    return datasets


def resample_data(
        data: pd.DataFrame,
        size: int) -> pd.DataFrame:
    data["original"] = True
    if len(data) == size:
        return data
    if len(data) > size:
        return data.sample(size)
    r = int(size / len(data))
    extra_rows = pd.concat(
        [data] * (r - 1) + [data.sample(size - r*len(data))],
        ignore_index=True)
    extra_rows["original"] = False
    data = pd.concat([data, extra_rows], ignore_index=True)

    return data


def apply_noise(
        data: pd.DataFrame,
        include_noise: List[str]) -> pd.DataFrame:
    data.loc[~data.original,
             include_noise] = data.loc[~data.original,
                                       include_noise].apply(_add_noise_row)
    return data


def _add_noise_row(row):
    sigma = 0.05*row.std()
    noise = np.random.normal(0, sigma, [len(row)])
    return row + noise
