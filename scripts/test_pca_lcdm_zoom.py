#!/usr/bin/env python3

"""Test PCA reconstruction errors for LCDM spectra."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from emu_like.datasets import Dataset
from emu_like.pca import PCA
from emu_like.scalers import Scaler


ROOT = '/data/emilio/emu_like'
MODEL = 'lcdm'
DATASET_RANGES = ['thin', 'std', 'ext']

OUTPUT_DIR = Path('/home/embellin/emu_like/output/test_pca_zoom')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SPECTRUM_CONFIGS = [
    ('pk_m', 'pk', 540, 600),
    ('pk_cb', 'pk', 540, 600),
    ('pk_weyl', 'pk', 540, 600),
    ('fk_m', 'pk', 540, 600),
    ('fk_cb', 'pk', 540, 600),
    ('fk_weyl', 'pk', 540, 600),
    ('cl_TT_lensed', 'cl', 300, 400),
    ('cl_TE_lensed', 'cl', 300, 400),
    ('cl_EE_lensed', 'cl', 300, 400),
    ('cl_BB_lensed', 'cl', 300, 400),
    ('cl_pp_lensed', 'cl', 300, 400),
    ('cl_Tp_lensed', 'cl', 300, 400),
]


def scale(y, y_train, y_test):
    """Scale the provided splits with a min-max scaler fitted on training data.

    Args:
        y (np.ndarray): Full dataset to scale.
        y_train (np.ndarray): Training subset used to fit the scaler.
        y_test (np.ndarray): Test subset scaled with the fitted scaler.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, Scaler]:
            Scaled full, training, and test arrays together with the
            fitted scaler instance.
    """

    y_scaler = Scaler.choose_one('MinMaxScaler')
    y_scaler.fit(y_train)

    y_all_scaled = y_scaler.transform(y)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    print('----> Done scale!')

    return y_all_scaled, y_train_scaled, y_test_scaled, y_scaler


def pca(n_components, y, y_train, y_test):
    """Fit PCA on the training data and transform each dataset split.

    Args:
        n_components (int): Number of PCA components to retain.
        y (np.ndarray): Full dataset to transform.
        y_train (np.ndarray): Training subset used to fit the PCA model.
        y_test (np.ndarray): Test subset to transform with the fitted PCA.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
            PCA-transformed full, training, and test arrays plus the
            fitted PCA wrapper.
    """

    y_pca = PCA(n_components=n_components)
    y_pca.fit(y_train)

    y_all_pca = y_pca.transform(y)
    y_train_pca = y_pca.transform(y_train)
    y_test_pca = y_pca.transform(y_test)
    print('----> Done PCA!')
    return y_all_pca, y_train_pca, y_test_pca, y_pca


def inv_pca(y_all_pca, y_train_pca, y_test_pca, y_pca):
    """Apply the inverse PCA transform to restore arrays to data space.

    Args:
        y_all_pca (np.ndarray): PCA-space representation of the full set.
        y_train_pca (np.ndarray): PCA-space representation of the training set.
        y_test_pca (np.ndarray): PCA-space representation of the test set.
        y_pca (PCA): Fitted PCA wrapper used for the inverse transform.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Reconstructed full, training, and test arrays in the original
            feature space.
    """

    y_all = y_pca.inverse_transform(y_all_pca)
    y_train = y_pca.inverse_transform(y_train_pca)
    y_test = y_pca.inverse_transform(y_test_pca)
    print('----> Done inverse PCA!')
    return y_all, y_train, y_test


def inv_scale(y_all_scaled, y_train_scaled, y_test_scaled, y_scaler):
    """Undo the scaling applied to each split using the fitted scaler.

    Args:
        y_all_scaled (np.ndarray): Scaled full dataset.
        y_train_scaled (np.ndarray): Scaled training subset.
        y_test_scaled (np.ndarray): Scaled test subset.
        y_scaler (Scaler): Fitted scaler used to reverse the transform.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Full, training, and test arrays restored to their original
            value ranges.
    """

    y_all = y_scaler.inverse_transform(y_all_scaled)
    y_train = y_scaler.inverse_transform(y_train_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)
    print('----> Done inverse scale!')
    return y_all, y_train, y_test


def select(y_all, y_train, y_test, pca_modes):
    """Keep only the selected PCA modes by zeroing every other column.

    Args:
        y_all (np.ndarray): PCA-space full dataset.
        y_train (np.ndarray): PCA-space training dataset.
        y_test (np.ndarray): PCA-space test dataset.
        pca_modes (Iterable[int]): Indices of the PCA modes to retain.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            PCA-space full, training, and test arrays containing only the
            selected modes.
    """

    tmp = np.zeros_like(y_all)
    tmp[:, pca_modes] = y_all[:, pca_modes]
    y_all = tmp
    tmp = np.zeros_like(y_train)
    tmp[:, pca_modes] = y_train[:, pca_modes]
    y_train = tmp
    tmp = np.zeros_like(y_test)
    tmp[:, pca_modes] = y_test[:, pca_modes]
    y_test = tmp
    return y_all, y_train, y_test


def diff(y_all, y_train, y_test, y_all_ref, y_train_ref, y_test_ref):
    """Compute mean absolute and relative errors against reference data.

    Args:
        y_all (np.ndarray): Reconstructed full dataset.
        y_train (np.ndarray): Reconstructed training dataset.
        y_test (np.ndarray): Reconstructed test dataset.
        y_all_ref (np.ndarray): Reference full dataset.
        y_train_ref (np.ndarray): Reference training dataset.
        y_test_ref (np.ndarray): Reference test dataset.

    Returns:
        dict[str, dict[str, np.ndarray]]:
            Nested mapping of split name to absolute and relative
            differences aggregated across samples.
    """

    diffs = {
        'all': {
            'rel': np.mean(np.abs(y_all / y_all_ref - 1.0), axis=0),
            'abs': np.mean(np.abs(y_all - y_all_ref), axis=0),
        },
        'train': {
            'rel': np.mean(np.abs(y_train / y_train_ref - 1.0), axis=0),
            'abs': np.mean(np.abs(y_train - y_train_ref), axis=0),
        },
        'test': {
            'rel': np.mean(np.abs(y_test / y_test_ref - 1.0), axis=0),
            'abs': np.mean(np.abs(y_test - y_test_ref), axis=0),
        },
    }
    print('----> Done diff!')
    return diffs


def get_modes_to_check(
    y_all,
    y_train,
    y_test,
    data,
    y_scaler,
    y_pca,
    min_mode,
    max_mode,
    n_modes_to_check=20,
):
    """Evaluate reconstruction errors while sweeping the number of modes.

    Args:
        y_all (np.ndarray): PCA-space full dataset.
        y_train (np.ndarray): PCA-space training dataset.
        y_test (np.ndarray): PCA-space test dataset.
        data (Dataset): Original dataset wrapper providing references.
        y_scaler (Scaler): Fitted scaler used for inverse scaling.
        y_pca (PCA): Fitted PCA wrapper used for reconstruction.
        min_mode (int): Minimum number of PCA modes to consider.
        max_mode (int): Maximum number of PCA modes to consider.
        n_modes_to_check (int, optional): Number of mode counts to sample
            between ``min_mode`` and ``max_mode``. Defaults to 20.

    Returns:
        dict[str, dict[str, dict[str, np.ndarray]]]:
            Error metrics keyed by split, metric type, and aggregation.
    """

    modes_to_check = np.linspace(
        min_mode,
        max_mode,
        num=n_modes_to_check,
        dtype=int,
    )

    # Init arrays
    diffs = {
        'x': modes_to_check,
    }
    for dataset in ['all', 'train', 'test']:
        diffs[dataset] = {}
        for type_diff in ['rel', 'abs']:
            diffs[dataset][type_diff] = {}
            for mean_or_max in ['mean', 'max']:
                diffs[dataset][type_diff][mean_or_max] = np.zeros(
                    len(modes_to_check),
                )

    for nmode, mode in enumerate(modes_to_check):
        y_all_tmp, y_train_tmp, y_test_tmp = select(
            y_all,
            y_train,
            y_test,
            range(mode),
        )

        y_all_tmp, y_train_tmp, y_test_tmp = inv_pca(
            y_all_tmp,
            y_train_tmp,
            y_test_tmp,
            y_pca,
        )
        y_all_tmp, y_train_tmp, y_test_tmp = inv_scale(
            y_all_tmp,
            y_train_tmp,
            y_test_tmp,
            y_scaler,
        )

        diffs_tmp = diff(
            y_all_tmp,
            y_train_tmp,
            y_test_tmp,
            data.y,
            data.y_train,
            data.y_test,
        )

        for dataset in ['all', 'train', 'test']:
            for type_diff in ['rel', 'abs']:
                diffs[dataset][type_diff]['mean'][nmode] = np.mean(
                    diffs_tmp[dataset][type_diff],
                )
                diffs[dataset][type_diff]['max'][nmode] = np.max(
                    diffs_tmp[dataset][type_diff],
                )

        print('Done mode {} ({}/{})'.format(mode, nmode + 1, n_modes_to_check))

    return diffs


def plot(diffs, y_pca, output_path, title):
    """Plot error curves and singular values for each dataset split.

    Args:
        diffs (dict[str, dict[str, dict[str, np.ndarray]]]):
            Error metrics to plot.
        y_pca (PCA): Fitted PCA wrapper providing singular values.
        output_path (Path | str): Destination path for the generated figure.
        title (str): Title to display on the plot grid.

    Returns:
        None: Creates the figure on disk and returns ``None``.
    """

    output_path = Path(output_path)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)

    for ndataset, dataset in enumerate(['all', 'train', 'test']):
        axs[0, ndataset].set_title('{} - rel_diff'.format(dataset))
        axs[1, ndataset].set_title('{} - abs_diff'.format(dataset))

        axs[0, ndataset].set_yscale('log')
        axs[1, ndataset].set_yscale('log')
        axs[0, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)
        axs[1, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)

        singular_values = (
            y_pca.pca.singular_values_ / y_pca.pca.singular_values_[0]
        )

        # First row, relative difference
        axs[0, ndataset].plot(
            diffs['x'],
            diffs[dataset]['rel']['mean'],
            label='mean',
        )
        axs[0, ndataset].plot(
            diffs['x'],
            diffs[dataset]['rel']['max'],
            label='max',
        )
        axs[0, ndataset].plot(
            singular_values * diffs[dataset]['rel']['mean'][0],
            label='singular values norm',
        )

        # Second row, absolute difference
        axs[1, ndataset].plot(
            diffs['x'],
            diffs[dataset]['abs']['mean'],
            label='mean',
        )
        axs[1, ndataset].plot(
            diffs['x'],
            diffs[dataset]['abs']['max'],
            label='max',
        )
        axs[1, ndataset].plot(
            singular_values * diffs[dataset]['abs']['mean'][0],
            label='singular values norm',
        )

    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    for spectrum, spectrum_type, min_mode, n_modes in SPECTRUM_CONFIGS:
        output_path = OUTPUT_DIR / f'{spectrum}_pca_errors.pdf'
        if output_path.exists():
            print(
                f'----> Skipping {spectrum}; output already present at '
                f'{output_path}'
            )
            continue

        data = [
            Dataset().load(
                path=os.path.join(
                    ROOT,
                    f'{MODEL}/sample/{spectrum_type}_100_{dr}.fits',
                ),
                name=spectrum,
                verbose=False,
            )
            for dr in DATASET_RANGES
        ]
        data = Dataset.join(data, verbose=True)

        data.train_test_split(0.9, 1543, verbose=True)

        y_all, y_train, y_test, y_scaler = scale(
            data.y, data.y_train, data.y_test
        )
        y_all, y_train, y_test, y_pca = pca(
            n_modes, y_all, y_train, y_test
        )

        diffs = get_modes_to_check(
            y_all,
            y_train,
            y_test,
            data,
            y_scaler,
            y_pca,
            min_mode,
            n_modes,
            n_modes_to_check=20,
        )

        plot(
            diffs,
            y_pca,
            OUTPUT_DIR / f'{spectrum}_pca_errors.pdf',
            f'{spectrum} PCA reconstruction errors',
        )
