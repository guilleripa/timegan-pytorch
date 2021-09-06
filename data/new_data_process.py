# %%
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


def data_preprocess(
    df: pd.DataFrame,
    max_seq_len: int,
    padding_value: float = -1.0,
    impute_method: str = "mode",
    scaling_method: str = "minmax",
    column_name: str = "apower",
    index_column: str = "datetime",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cargar los datos.
    hacerlos un coso 3d
    filas: secuencias
    columna: timestamp
    profundidad: columnas del dataset
    """
    # selecciono la columna
    # dado un timestamp, juntame todo

    if not index_column and index_column not in df:
        raise ValueError(f"{index_column} not in df")

    if column_name in df:
        df = df[[index_column, column_name]]
    else:
        raise ValueError(f"{column_name} not in df")


# %%
DATA_PATH = Path("/Volumes/GuilleSSD/edc_data")
df = pd.read_csv(DATA_PATH / "appliance_consumption_data.csv", parse_dates=["datetime"])
appliances = pd.read_csv(DATA_PATH / "appliances.csv")

# %%
df = df[["datetime", "meter_id", "apower"]]
# yo lo que quiero es agrupar por meter primero, después cada meter cortarlo en intervalos
# %%
df = df.dropna(subset=["apower"])
test = df[df["meter_id"] == "00124B0002CBABF1"]
# %%
# %%
# tengo un meter solo, quiero que me devuelvas rows con intervalos


def get_intervals(
    df: pd.DataFrame,
    seq_length: int,
    column_name: str,
    offset: DateOffset = DateOffset(minutes=30),
    error: DateOffset = DateOffset(seconds=10),
) -> np.array:
    next_datetime = df["datetime"].shift(-1)

    series = None
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
            if series is None:
                series = interval
            else:
                series = np.concatenate((series, interval), axis=0)

    return series


X_test = get_intervals(test, 30, "apower")
# %%
X_test.shape
# %%
selected_customers = [170001, 170004, 170005, 170006, 170008]
customer_meters = appliances[appliances["customer_id"].isin(selected_customers)]
appliance_meters = customer_meters[customer_meters["appl_type"] != "site meter"]
data = df[df["meter_id"].isin(appliance_meters["meter_id"])]

column_name = "apower"
seq_length = 30  # x cantidad de minutos
X = None
for meter_id in appliance_meters["meter_id"].unique():
    if len(data[data["meter_id"] == meter_id]) != 0:

        X_meter = get_intervals(
            data[data["meter_id"] == meter_id], seq_length, column_name
        )
        if X is None:
            X = X_meter
        else:
            X = np.stack((X, X_meter), axis=0)

# %%
