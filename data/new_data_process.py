from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_PATH = Path("/Volumes/GuilleSSD/edc_data")


def load_data(
    appliance_consumption_path: Union[Path, str],
    metadata_path: Union[Path, str],
    merge_on: str = "meter_id",
    parse_dates: list = ["datetime"],
    selected_customers: list = [170001, 170004, 170005, 170006, 170008],
) -> pd.DataFrame:
    df = pd.read_csv(appliance_consumption_path, parse_dates=parse_dates)
    metadata = pd.read_csv(metadata_path)

    if selected_customers:
        customer_meters = metadata[metadata["customer_id"].isin(selected_customers)]
        appliance_meters = customer_meters[customer_meters["appl_type"] != "site meter"]
        df = df[df["meter_id"].isin(appliance_meters["meter_id"])]

    return df.merge(metadata, on=merge_on)


def data_preprocess(
    df: pd.DataFrame,
    max_seq_len: int,
    scaling_method: str = "minmax",
    column_name: str = "apower",
    index_column: str = "datetime",
    dropna: bool = True,
) -> np.ndarray:

    if not index_column and index_column not in df:
        raise ValueError(f"{index_column} not in df")

    if column_name not in df:
        raise ValueError(f"{column_name} not in df")

    if dropna:
        df = df.dropna(subset=[column_name])

    X = create_intervals(df, max_seq_len, column_name)

    return X


def create_intervals(
    df: pd.DataFrame, max_seq_len: int, column_name: str
) -> np.ndarray:
    X = np.array([])
    for meter_id in df["meter_id"].unique():
        if len(df[df["meter_id"] == meter_id]) != 0:

            X_meter = create_meter_intervals(
                df[df["meter_id"] == meter_id], max_seq_len, column_name
            )
            if X.size == 0:
                X = X_meter
            else:
                X = np.concatenate((X, X_meter), axis=0)

    return X


def create_meter_intervals(
    df: pd.DataFrame,
    seq_length: int,
    column_name: str,
    offset: DateOffset = DateOffset(minutes=30),
    error: DateOffset = DateOffset(seconds=10),
) -> np.ndarray:
    next_datetime = df["datetime"].shift(-1)

    series = np.array([])
    current_date = df["datetime"].iloc[0]

    while current_date:
        interval = df[
            (current_date <= df["datetime"])
            & (df["datetime"] < current_date + offset + error)
        ][:seq_length]

        if interval.empty:
            break
        else:
            current_date = next_datetime[interval.index].iloc[-1]

        if len(interval) == seq_length:
            interval = np.array([interval[column_name]])
            if series.size == 0:
                series = interval
            else:
                series = np.concatenate((series, interval), axis=0)

    return series
