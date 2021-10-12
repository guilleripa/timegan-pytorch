# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data.new_data_process import data_preprocess, load_data
from matplotlib import pyplot as plt
from models.timegan import TimeGAN

import research.start  # noqa

DATA_PATH = Path("/Volumes/GuilleSSD/edc_data")
# %%
df = load_data(
    DATA_PATH / "appliance_consumption_data.csv", DATA_PATH / "appliances.csv"
)
X = data_preprocess(df, 30, cyclic_features=False)
# X = data_preprocess(df, 30)

# %%
X.shape
# %%
# Cargar modelo y generar varias de media hora y varios de todo el día.
model_dir = "300noCt75"
args = torch.load(
    DATA_PATH / "models" / model_dir / "args.pickle", map_location=torch.device("cpu")
)
args.device = torch.device("cpu", index=0)
weights = torch.load(
    DATA_PATH / "models" / model_dir / "model.pt", map_location=torch.device("cpu")
)
model = TimeGAN(args)
model.load_state_dict(weights)
model.to(args.device)
model.eval()
model
# %%
examples = 100
seq_len = args.max_seq_len
T = [seq_len] * examples
with torch.no_grad():
    # Generate fake data
    Z = torch.rand(examples, args.max_seq_len, args.Z_dim)

    generated_data = model(X=None, T=T, Z=Z, obj="inference")
    if args.Z_dim == 5:
        # sort
        sen_data = 2 * generated_data[:, 0, -2] - 1
        cos_data = np.clip(generated_data[:, 0, -1] * 2 - 1, -1, 1)
        generated_data = generated_data[
            np.argsort(
                np.where(
                    sen_data >= 0, np.arccos(cos_data), 2 * np.pi - np.arccos(cos_data)
                )
            )
        ]
# %%
generated_data
# %% [markdown]
# Ya tengo la data generada, ahora la quiero plottear.
# Van todas juntas en una única gráfica.
# %%
appliances_list = [
    "fridge",
    "electric water heating appliance",
    "air conditioner",
]
# %%
def get_time(sin, cos):
    max_seconds = 60 * 60 * 24
    total = 0
    if sin < 0:
        total += np.pi
    if cos < 0:
        total += np.pi / 2


# %%
models_dirs = (DATA_PATH / "models").glob("300*/")
for model_dir in models_dirs:
    args = torch.load(
        DATA_PATH / "models" / model_dir / "args.pickle",
        map_location=torch.device("cpu"),
    )
    args.device = torch.device("cpu", index=0)
    print(model_dir.name)
    print("\tcyclical:", args.Z_dim == 5)
    print("\tepochs:", args.emb_epochs)
    print("\tlen:", args.max_seq_len)
# %%
def show_samples(generated_data, model_args, num_samples=5, save_fig=True):
    random_examples = np.random.choice(len(generated_data), num_samples, replace=False)
    fig, axes = plt.subplots(
        1,
        num_samples,
        figsize=(num_samples * 3, 3),
    )
    axes = np.array([axes]).flatten()
    for ax, sample_idx in zip(axes, random_examples):
        lgd = fig.legend(
            appliances_list,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            shadow=True,
            ncol=3,
        )
        for appliance in range(3):
            ax.plot(generated_data[sample_idx, :, appliance])
        suptitle = fig.suptitle(
            "Generated samples from TimeGAN - "
            f"{{epochs: {model_args.emb_epochs}, "
            f"seq len:{model_args.max_seq_len}, "
            f"has cyclic: {'yes' if model_args.Z_dim == 5 else 'no'}}}",
            wrap=True,
        )
    fig.tight_layout()

    if save_fig:
        title = (
            f"{model_args.emb_epochs}len{model_args.max_seq_len}"
            f"{'yesC' if model_args.Z_dim == 5 else 'noC'}"
        )
        fig.savefig(
            DATA_PATH / "figs" / title,
            bbox_extra_artists=(lgd, suptitle),
            bbox_inches="tight",
        )
    return fig


def load_model(model_dir):
    args = torch.load(
        DATA_PATH / "models" / model_dir / "args.pickle",
        map_location=torch.device("cpu"),
    )
    args.device = torch.device("cpu", index=0)
    weights = torch.load(
        DATA_PATH / "models" / model_dir / "model.pt", map_location=torch.device("cpu")
    )
    model = TimeGAN(args)
    model.load_state_dict(weights)
    model.to(args.device)
    model.eval()

    return model, args


def generate_data(model, args, samples=100):
    seq_len = args.max_seq_len
    T = [seq_len] * samples
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand(samples, seq_len, args.Z_dim)

        generated_data = model(X=None, T=T, Z=Z, obj="inference")
        if args.Z_dim == 5:
            # sort
            sen_data = 2 * generated_data[:, 0, -2] - 1
            cos_data = np.clip(generated_data[:, 0, -1] * 2 - 1, -1, 1)
            generated_data = generated_data[
                np.argsort(
                    np.where(
                        sen_data >= 0,
                        np.arccos(cos_data),
                        2 * np.pi - np.arccos(cos_data),
                    )
                )
            ]

    return generated_data


# %%
models_dirs = (DATA_PATH / "models").glob("[!.]*/")
generated_samples = 100
fig_samples = 5
for model_dir in models_dirs:
    model, args = load_model(model_dir)
    generated_data = generate_data(model, args, samples=generated_samples)
    fig = show_samples(generated_data, args, num_samples=fig_samples)
# %%
X
# %%
num_samples = 5
random_examples = np.random.choice(len(X), num_samples, replace=False)
fig, axes = plt.subplots(
    1,
    num_samples,
    figsize=(num_samples * 3, 3),
)
axes = np.array([axes]).flatten()
for ax, sample_idx in zip(axes, random_examples):
    lgd = fig.legend(
        appliances_list,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        shadow=True,
        ncol=3,
    )
    for appliance in range(3):
        ax.plot(X[sample_idx, :, appliance])
    suptitle = fig.suptitle(
        "Real samples from EDC-UY dataset",
        wrap=True,
    )
fig.tight_layout()

# title = (
#     f"{model_args.emb_epochs}len{model_args.max_seq_len}"
#     f"{'yesC' if model_args.Z_dim == 5 else 'noC'}"
# )
# fig.savefig(
#     DATA_PATH / "figs" / title,
#     bbox_extra_artists=(lgd, suptitle),
#     bbox_inches="tight",
# )

# %%
