# %%
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from data.new_data_process import (
    create_customer_intervals,
    create_intervals,
    data_preprocess,
    load_data,
    remove_outliers,
    select_appliances,
)
from matplotlib import pyplot as plt
from scipy import stats

import research.start  # noqa

DATA_PATH = Path("/Volumes/GuilleSSD/edc_data")

# %%
df = load_data(
    DATA_PATH / "appliance_consumption_data.csv", DATA_PATH / "appliances.csv"
)
# %%
ax = df["apower"].plot(figsize=(500, 20))
# %%
X = data_preprocess(df, 30)
# %%
plt.bar(np.arange(len(X.flatten())), X.flatten())
# %%
pere = pd.DataFrame(X.flatten().T)

# %%
df
# %%
df = remove_outliers(df, "apower")
# %%
df
# %%
pere.plot(figsize=(500, 20))
# %%
df
# %%
appliances_list = [
    "fridge",
    "electric water heating appliance",
    "air conditioner",
]
# %%
df.groupby("customer_id")["appl_type"].value_counts()
# %%
df[df["appl_type"].isin(appliances_list)].groupby(["customer_id", "meter_id"])[
    "appl_type"
].value_counts()
# %%
appliances = pd.read_csv(DATA_PATH / "appliances.csv")
selected_users = [170001, 170004, 170005, 170006, 170008]
appliances = appliances[appliances["customer_id"].isin(selected_users)]
# %%
appliances[appliances["appl_type"] != "site meter"].groupby(
    ["customer_id", "appl_type"]
)["meter_id"].count()
# %%
df = df[df["appl_type"].isin(appliances_list) & df["customer_id"].isin(selected_users)]

# %%
pere = df.groupby("customer_id").apply(
    lambda df: df.pivot(index="datetime", columns=["meter_id"], values=["apower"])
)
# %%
pere
# %%
from datetime import timedelta

delta = timedelta(
    days=0,
    seconds=30,
    microseconds=0,
    milliseconds=0,
    minutes=0,
    hours=0,
    weeks=0,
)
# %%
df[(df["datetime"].sort_values().diff() > delta)]

# %%
(pere.reset_index().dropna()["datetime"].sort_values().diff() < delta).sum()
# %%
# %%
# entonces, elijo los users, elijo los meters y hago groupby pivot por appl_type
df = load_data(
    DATA_PATH / "appliance_consumption_data.csv", DATA_PATH / "appliances.csv"
)
selected_users = [170001, 170004, 170005, 170006, 170008]
appliances_list = [
    "fridge",
    "electric water heating appliance",
    "air conditioner",
]
df = df[df["appl_type"].isin(appliances_list) & df["customer_id"].isin(selected_users)]

# %%
def get_max_consumption_meter(column_name: str, df: pd.DataFrame):
    max_meter = df.groupby("meter_id")[column_name].sum().idxmax()
    return df[df["meter_id"] == max_meter]


from functools import partial

get_max_meter = partial(get_max_consumption_meter, "apower")

# %%
# Conseguir el appliance con más gasto entre los repetidos
df = df.groupby(["customer_id", "appl_type"]).apply(get_max_consumption_meter)
# drop groupby axes
df = df.droplevel([0, 1])
# %%
pere = (
    df.groupby("customer_id")
    .apply(
        lambda df: df.pivot(index="datetime", columns=["appl_type"], values=["apower"])
    )["apower"]
    .reset_index()
)
# %%
df_time = pere["datetime"]
day_seconds = (df_time.dt.hour * 60 + df_time.dt.minute) * 60 + df_time.dt.second
# %%
seconds_in_day = 24 * 60 * 60
pere["sin_time"] = np.sin(2 * np.pi * day_seconds / seconds_in_day)
pere["cos_time"] = np.cos(2 * np.pi * day_seconds / seconds_in_day)
# %%
pere.plot.scatter("sin_time", "cos_time").set_aspect("equal")
# %%
pere = pere.fillna(0)
pere[appliances_list + ["sin_time", "cos_time"]].to_numpy()
# %%
column_name = "apower"
df = select_appliances(df, appliances_list)

df = remove_outliers(df, column_name)

# %%
df
# %%
df1 = create_intervals(df, "all", 30, column_name)
# %%
df2 = create_customer_intervals(df, appliances_list, column_name)
# %%
columns = appliances_list + ["fridge", "sin_time", "cos_time"]
np.expand_dims(df2[columns].to_numpy(), -1)[0]
# %%
np.array([df2["sin_time"]])
# %%
np.array([df2[columns][:30].to_numpy()]).shape
# %%
np.array([df2["air conditioner"][:30]]).shape
# %%
df = load_data(
    DATA_PATH / "appliance_consumption_data.csv", DATA_PATH / "appliances.csv"
)
appliances_list = [
    "fridge",
    "electric water heating appliance",
    "air conditioner",
]
X = data_preprocess(df, 30, cyclic_features=False)
# X = data_preprocess(df, 30)

# %%
X.shape
