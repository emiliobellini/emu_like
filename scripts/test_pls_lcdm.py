#!/usr/bin/env python3

"""Evaluate Partial Least Squares reconstructions for LCDM spectra."""

import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from emu_like.datasets import Dataset
from emu_like.scalers import Scaler


ROOT = "/data/emilio/emu_like"
MODEL = "lcdm"
DATASET_RANGES = ["thin", "std", "ext"]

OUTPUT_DIR = Path("/home/embellin/emu_like/output/test_pls_lcdm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SPECTRUM_CONFIGS = [
    {
        "spectrum": "pk_m",
        "spectrum_type": "pk",
        "min_components": 16,
        "n_components_to_check": 8,
        "zoom_min_fraction": 0.75,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "LogStandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 1,
    },
    {
        "spectrum": "pk_cb",
        "spectrum_type": "pk",
        "min_components": 16,
        "n_components_to_check": 8,
        "zoom_min_fraction": 0.75,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "pk_weyl",
        "spectrum_type": "pk",
        "min_components": 16,
        "n_components_to_check": 8,
        "zoom_min_fraction": 0.75,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_TT_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_TE_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_EE_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_BB_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_pp_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
    {
        "spectrum": "cl_Tp_lensed",
        "spectrum_type": "cl",
        "min_components": 32,
        "n_components_to_check": 7,
        "zoom_min_fraction": 0.85,
        "zoom_n_components_to_check": 5,
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "max_iter": 500,
        "tol": 1e-06,
        "verbose": 0,
    },
]


def scale_xy(
    x_all: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_all: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_scaler_name: str,
    y_scaler_name: str,
):
    """Scale train/test/all splits for x and y, returning scaled arrays and scalers."""

    x_scaler = Scaler.choose_one(x_scaler_name)
    x_scaler.fit(x_train)
    x_all_scaled = x_scaler.transform(x_all)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = Scaler.choose_one(y_scaler_name)
    y_scaler.fit(y_train)
    y_all_scaled = y_scaler.transform(y_all)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return (
        x_all_scaled,
        x_train_scaled,
        x_test_scaled,
        y_all_scaled,
        y_train_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )


def safe_relative_error(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute |values/reference - 1| guarding against division by zero."""

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(values, reference, out=np.ones_like(values), where=reference != 0.0)
    return np.abs(ratio - 1.0)


def diff(
    x_all_pred: np.ndarray,
    x_train_pred: np.ndarray,
    x_test_pred: np.ndarray,
    x_all_ref: np.ndarray,
    x_train_ref: np.ndarray,
    x_test_ref: np.ndarray,
):
    """Compute mean and max absolute/relative differences for each split."""

    diffs = {
        "all": {},
        "train": {},
        "test": {},
    }

    for key, pred, ref in [
        ("all", x_all_pred, x_all_ref),
        ("train", x_train_pred, x_train_ref),
        ("test", x_test_pred, x_test_ref),
    ]:
        rel = safe_relative_error(pred, ref)
        abs_diff = np.abs(pred - ref)
        diffs[key] = {
            "rel": {
                "mean": np.mean(rel, axis=0),
                "max": np.max(rel, axis=0),
            },
            "abs": {
                "mean": np.mean(abs_diff, axis=0),
                "max": np.max(abs_diff, axis=0),
            },
        }

    return diffs


def get_component_grid(min_components: int, max_components: int, n_components: int) -> np.ndarray:
    """Create a sorted, unique array of component counts to evaluate."""

    components = np.linspace(min_components, max_components, num=n_components, dtype=int)
    components = np.unique(components)
    components = components[components > 0]
    return components


def evaluate_pls_components(
    x_all_scaled: np.ndarray,
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
    y_all_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    x_all: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_all: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    components: np.ndarray,
    x_scaler,
    y_scaler,
    *,
    max_iter: int,
    tol: float,
    verbose: int,
):
    """Fit PLS models for different component counts and collect reconstruction diffs."""

    diffs_y = {"x": components}
    diffs_x = {"x": components}
    for dataset_key in ["all", "train", "test"]:
        diffs_y[dataset_key] = {
            "rel": {
                "mean": np.zeros_like(components, dtype=float),
                "max": np.zeros_like(components, dtype=float),
            },
            "abs": {
                "mean": np.zeros_like(components, dtype=float),
                "max": np.zeros_like(components, dtype=float),
            },
        }
        diffs_x[dataset_key] = {
            "rel": {
                "mean": np.zeros_like(components, dtype=float),
                "max": np.zeros_like(components, dtype=float),
            },
            "abs": {
                "mean": np.zeros_like(components, dtype=float),
                "max": np.zeros_like(components, dtype=float),
            },
        }

    histories = {}

    for idx, n_components in enumerate(components):
        start = perf_counter()
        pls = PLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=max_iter,
            tol=tol,
        )
        pls.fit(x_train_scaled, y_train_scaled)
        elapsed = perf_counter() - start

        y_all_pred_scaled = pls.predict(x_all_scaled)
        y_train_pred_scaled = pls.predict(x_train_scaled)
        y_test_pred_scaled = pls.predict(x_test_scaled)

        y_all_pred = y_scaler.inverse_transform(y_all_pred_scaled)
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
        y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

        x_all_scores = pls.transform(x_all_scaled)
        x_train_scores = pls.transform(x_train_scaled)
        x_test_scores = pls.transform(x_test_scaled)
        x_all_recon_scaled = pls.inverse_transform(x_all_scores)
        x_train_recon_scaled = pls.inverse_transform(x_train_scores)
        x_test_recon_scaled = pls.inverse_transform(x_test_scores)
        x_all_recon = x_scaler.inverse_transform(x_all_recon_scaled)
        x_train_recon = x_scaler.inverse_transform(x_train_recon_scaled)
        x_test_recon = x_scaler.inverse_transform(x_test_recon_scaled)

        diffs_y_tmp = diff(
            y_all_pred,
            y_train_pred,
            y_test_pred,
            y_all,
            y_train,
            y_test,
        )
        diffs_x_tmp = diff(
            x_all_recon,
            x_train_recon,
            x_test_recon,
            x_all,
            x_train,
            x_test,
        )

        for dataset_key in ["all", "train", "test"]:
            for metric in ["rel", "abs"]:
                diffs_y[dataset_key][metric]["mean"][idx] = diffs_y_tmp[dataset_key][metric]["mean"]
                diffs_y[dataset_key][metric]["max"][idx] = diffs_y_tmp[dataset_key][metric]["max"]
                diffs_x[dataset_key][metric]["mean"][idx] = diffs_x_tmp[dataset_key][metric]["mean"]
                diffs_x[dataset_key][metric]["max"][idx] = diffs_x_tmp[dataset_key][metric]["max"]

        histories[int(n_components)] = {"fit_time": elapsed}
        if verbose:
            print(
                f"Done components {n_components} ({idx + 1}/{len(components)}) "
                f"in {elapsed:.2f} s"
            )

    return diffs_y, diffs_x, histories


def plot_diffs(diffs: dict, *, base_name: str, title: str):
    """Plot the mean/max absolute and relative differences for each dataset split."""

    diff_path = OUTPUT_DIR / f"{base_name}.pdf"
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)

    for ndataset, dataset in enumerate(["all", "train", "test"]):
        axs[0, ndataset].set_title(f"{dataset} - rel_diff")
        axs[1, ndataset].set_title(f"{dataset} - abs_diff")

        axs[0, ndataset].set_yscale("log")
        axs[1, ndataset].set_yscale("log")
        axs[0, ndataset].set_xlim(diffs["x"][0] - 1, diffs["x"][-1] + 1)
        axs[1, ndataset].set_xlim(diffs["x"][0] - 1, diffs["x"][-1] + 1)

        axs[0, ndataset].plot(diffs["x"], diffs[dataset]["rel"]["mean"], label="mean")
        axs[0, ndataset].plot(diffs["x"], diffs[dataset]["rel"]["max"], label="max")

        axs[1, ndataset].plot(diffs["x"], diffs[dataset]["abs"]["mean"], label="mean")
        axs[1, ndataset].plot(diffs["x"], diffs[dataset]["abs"]["max"], label="max")

    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(diff_path, dpi=150)
    plt.close(fig)

    return diff_path


def main():
    for config in SPECTRUM_CONFIGS:
        spectrum = config["spectrum"]
        spectrum_type = config["spectrum_type"]
        verbose = config.get("verbose", 0)

        if verbose:
            print(f"Processing spectrum {spectrum}")

        data = [
            Dataset().load(
                path=os.path.join(
                    ROOT,
                    f"{MODEL}/sample/{spectrum_type}_100_{dr}.fits",
                ),
                name=spectrum,
                verbose=False,
            )
            for dr in DATASET_RANGES
        ]
        data = Dataset.join(data, verbose=verbose > 0)
        data.train_test_split(0.9, 1543, verbose=verbose > 0)

        x_all = data.x.copy()
        x_train = data.x_train.copy()
        x_test = data.x_test.copy()
        y_all = data.y.copy()
        y_train = data.y_train.copy()
        y_test = data.y_test.copy()

        (
            x_all_scaled,
            x_train_scaled,
            x_test_scaled,
            y_all_scaled,
            y_train_scaled,
            y_test_scaled,
            x_scaler,
            y_scaler,
        ) = scale_xy(
            x_all,
            x_train,
            x_test,
            y_all,
            y_train,
            y_test,
            config["x_scaler"],
            config["y_scaler"],
        )

        max_components_allowed = min(
            x_train_scaled.shape[0],
            x_train_scaled.shape[1],
            y_train_scaled.shape[1],
        )
        min_components = min(config["min_components"], max_components_allowed)
        max_components = max_components_allowed

        components = get_component_grid(
            min_components,
            max_components,
            config["n_components_to_check"],
        )

        diffs_y, diffs_x, histories = evaluate_pls_components(
            x_all_scaled,
            x_train_scaled,
            x_test_scaled,
            y_all_scaled,
            y_train_scaled,
            y_test_scaled,
            x_all,
            x_train,
            x_test,
            y_all,
            y_train,
            y_test,
            components,
            x_scaler,
            y_scaler,
            max_iter=config["max_iter"],
            tol=config["tol"],
            verbose=verbose,
        )

        title = f"{spectrum} PLS reconstruction errors"
        plot_diffs(diffs_y, base_name=f"{spectrum}_pls_y_diffs", title=title)
        plot_diffs(
            diffs_x,
            base_name=f"{spectrum}_pls_x_diffs",
            title=f"{spectrum} PLS x recon errors",
        )

        if verbose:
            for n_comp, info in histories.items():
                print(f"Components {n_comp}: fit_time={info['fit_time']:.2f}s")


if __name__ == "__main__":
    main()
