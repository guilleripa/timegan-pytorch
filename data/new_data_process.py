from functools import partial
from pathlib import Path
from typing import Union

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
    appliances: Union[list, str] = [
        "fridge",
        "air conditioner",
        "electric water heating appliance",
    ],
    cyclic_features: bool = True,
) -> np.ndarray:

    if not index_column and index_column not in df:
        raise ValueError(f"{index_column} not in df")

    if column_name not in df:
        raise ValueError(f"{column_name} not in df")

    if dropna:
        df = df.dropna(subset=[column_name])

    df = select_appliances(df, appliances)

    df = remove_outliers(df, column_name)

    X = create_intervals(
        df, appliances, max_seq_len, column_name, cyclic_features=cyclic_features
    )

    X = scale_intervals(X, scaling_method)

    if len(X.shape) <= 2:
        X = np.expand_dims(X, -1)

    return X


def remove_outliers(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    original_len = len(df)
    z_scores = stats.zscore(df[column_name], nan_policy="omit")
    z_filter = np.abs(z_scores) < 3
    df = df[z_filter]
    print(f"Dropped {original_len - len(df)} rows (outliers)\n")

    return df


def select_appliances(df, appliances):
    if appliances == "all":
        return df

    set_difference = np.setdiff1d(appliances, df["appl_type"].unique())

    if set_difference.size > 0:
        raise KeyError(f"{set_difference} are not in df.")

    if type(appliances) == list:
        df = df[df["appl_type"].isin(appliances)]

    return df


def create_intervals(
    df: pd.DataFrame,
    appliances: Union[list, str],
    max_seq_len: int,
    column_name: str,
    cyclic_features: bool = True,
) -> np.ndarray:
    X = np.array([])

    if appliances == "all":
        index_name = "meter_id"
        columns = column_name
    else:
        df = create_customer_intervals(df, column_name, cyclic_features=cyclic_features)
        columns = appliances
        if cyclic_features:
            columns += ["sin_time", "cos_time"]
        index_name = "customer_id"

    for meter_id in df[index_name].unique():
        X_index = create_intervals_(
            df[df[index_name] == meter_id], max_seq_len, columns
        )
        if X.size == 0:
            X = X_index
        else:
            X = np.concatenate((X, X_index), axis=0)

    return X


def create_customer_intervals(
    df: pd.DataFrame,
    column_name: str,
    cyclic_features: bool = True,
) -> pd.DataFrame:

    get_max_meter = partial(get_max_consumption_meter, column_name)
    # Conseguir el appliance con más gasto entre los repetidos
    df = df.groupby(["customer_id", "appl_type"]).apply(get_max_meter)
    # drop groupby axes
    df = df.droplevel([0, 1])

    new_df = (
        df.groupby("customer_id")
        .apply(
            lambda df: df.pivot(
                index="datetime", columns=["appl_type"], values=["apower"]
            )
        )["apower"]
        .reset_index()
    )

    if cyclic_features:
        df_time = new_df["datetime"]
        day_seconds = (
            df_time.dt.hour * 60 + df_time.dt.minute
        ) * 60 + df_time.dt.second

        seconds_in_day = 24 * 60 * 60
        new_df["sin_time"] = np.sin(2 * np.pi * day_seconds / seconds_in_day)
        new_df["cos_time"] = np.cos(2 * np.pi * day_seconds / seconds_in_day)

    new_df = new_df.fillna(0)

    return new_df


def create_intervals_(
    df: pd.DataFrame,
    seq_length: int,
    columns: Union[str, list],
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
            interval = np.array([interval[columns]])
            if series.size == 0:
                series = interval
            else:
                series = np.concatenate((series, interval), axis=0)

    return series


def scale_intervals(X: np.ndarray, scaling_method: str) -> np.ndarray:
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "standard":
        scaler = StandardScaler()

    if len(X.shape) <= 2:
        X_transformed = scaler.fit_transform(X)
    else:
        X_transformed = np.empty(X.T.shape)
        for i, appliance_readings in enumerate(X.T):
            X_transformed_i = scaler.fit_transform(appliance_readings)
            X_transformed[i] = X_transformed_i
        X_transformed = X_transformed.T

    return X_transformed


def get_max_consumption_meter(column_name: str, df: pd.DataFrame):
    max_meter = df.groupby("meter_id")[column_name].sum().idxmax()
    return df[df["meter_id"] == max_meter]
