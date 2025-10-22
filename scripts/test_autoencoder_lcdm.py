#!/usr/bin/env python3

"""Test autoencoder reconstruction errors for LCDM spectra."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from time import perf_counter

from emu_like.datasets import Dataset
from emu_like.scalers import Scaler


ROOT = '/data/emilio/emu_like'
MODEL = 'lcdm'
DATASET_RANGES = ['thin', 'std', 'ext']

OUTPUT_DIR = Path('/home/embellin/emu_like/output/test_autoencoder_lcdm')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECTRUM_CONFIGS = [
    {
        'spectrum': 'pk_m',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 1,
    },
    {
        'spectrum': 'pk_cb',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'pk_weyl',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'fk_m',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'fk_cb',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'fk_weyl',
        'spectrum_type': 'pk',
        'min_latent': 16,
        'n_latent_to_check': 8,
        'zoom_min_latent_fraction': 0.75,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (1024, 800, 600),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_tt_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_te_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_ee_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_bb_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_pp_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
    {
        'spectrum': 'cl_tp_lensed',
        'spectrum_type': 'cl',
        'min_latent': 32,
        'n_latent_to_check': 7,
        'zoom_min_latent_fraction': 0.85,
        'zoom_n_latent_to_check': 5,
        'epochs': 200,
        'batch_size': 128,
        'patience': 20,
        'learning_rate': 1e-3,
        'hidden_units': (5000, 4000, 2999),
        'activation': 'relu',
        'verbose': 0,
    },
]

np.random.seed(0)
tf.random.set_seed(0)


def scale(y, y_train, y_test, scaler_name='MinMaxCommonScaler'):
    """Scale the provided splits with a scaler fitted on the training data.

    Args:
        y (np.ndarray): Full dataset to scale.
        y_train (np.ndarray): Training subset used to fit the scaler.
        y_test (np.ndarray): Test subset scaled with the fitted scaler.
        scaler_name (str): Identifier passed to ``Scaler.choose_one``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, Scaler]:
            Scaled full, training, and test arrays together with the
            fitted scaler instance.
    """

    y_scaler = Scaler.choose_one(scaler_name)
    y_scaler.fit(y_train)

    y_all_scaled = y_scaler.transform(y)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    print('----> Done scale!')

    return y_all_scaled, y_train_scaled, y_test_scaled, y_scaler


def build_autoencoder(
    input_dim,
    latent_dim,
    hidden_units=(256, 128),
    activation='relu',
):
    """Create a symmetric fully connected autoencoder architecture.

    Args:
        input_dim (int): Number of features in the input vectors.
        latent_dim (int): Size of the bottleneck latent space.
        hidden_units (tuple[int, ...]): Dense layer widths for the encoder.
        activation (str): Activation applied to the hidden layers.

    Returns:
        keras.Model: Autoencoder model ready for training.
    """

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
    latent = layers.Dense(latent_dim, name='latent')(x)

    x = latent
    for units in reversed(hidden_units):
        x = layers.Dense(units, activation=activation)(x)
    outputs = layers.Dense(input_dim)(x)

    autoencoder = keras.Model(
        inputs,
        outputs,
        name=f'autoencoder_{latent_dim}',
    )
    return autoencoder


def train_autoencoder(
    y_train,
    y_val,
    latent_dim,
    *,
    epochs=200,
    batch_size=128,
    patience=20,
    hidden_units=(256, 128),
    activation='relu',
    learning_rate=1e-3,
    verbose=0,
):
    """Train an autoencoder and return the fitted model and training history.

    Args:
        y_train (np.ndarray): Training dataset used for optimisation.
        y_val (np.ndarray | None): Validation dataset for early stopping.
        latent_dim (int): Bottleneck dimensionality to train.
        epochs (int): Maximum number of training epochs. Defaults to 200.
        batch_size (int): Minibatch size. Defaults to 128.
        patience (int): Epoch patience for the early stopping callback.
        hidden_units (tuple[int, ...]): Dense layer widths for encoder/decoder.
        activation (str): Activation applied to the hidden layers.
        learning_rate (float): Adam optimiser learning rate.
        verbose (int): Keras verbosity flag passed to ``fit``.

    Returns:
        tuple[keras.Model, dict[str, list[float]]]:
            Trained autoencoder model and the recorded loss history.
    """

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32) if y_val is not None else None

    autoencoder = build_autoencoder(
        y_train.shape[1],
        latent_dim,
        hidden_units=hidden_units,
        activation=activation,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    callbacks = []
    if y_val is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
            )
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(1, patience // 2),
                verbose=verbose,
            )
        )
    else:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=patience,
                restore_best_weights=True,
            )
        )

    history = autoencoder.fit(
        y_train,
        y_train,
        validation_data=(y_val, y_val) if y_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    print('----> Done autoencoder training!')
    return autoencoder, history.history


def inv_scale(y_all_scaled, y_train_scaled, y_test_scaled, y_scaler):
    """Undo the scaling applied to each split using the fitted scaler.

    Args:
        y_all_scaled (np.ndarray): Scaled full dataset.
        y_train_scaled (np.ndarray): Scaled training dataset.
        y_test_scaled (np.ndarray): Scaled test dataset.
        y_scaler (Scaler): Fitted scaler to reverse the transformation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            Full, training, and test arrays restored to original ranges.
    """

    y_all = y_scaler.inverse_transform(y_all_scaled)
    y_train = y_scaler.inverse_transform(y_train_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)
    print('----> Done inverse scale!')
    return y_all, y_train, y_test


def diff(y_all, y_train, y_test, y_all_ref, y_train_ref, y_test_ref):
    """Compute mean and maximum absolute and relative differences.

    Args:
        y_all (np.ndarray): Reconstructed full dataset.
        y_train (np.ndarray): Reconstructed training dataset.
        y_test (np.ndarray): Reconstructed test dataset.
        y_all_ref (np.ndarray): Reference full dataset.
        y_train_ref (np.ndarray): Reference training dataset.
        y_test_ref (np.ndarray): Reference test dataset.

    Returns:
        dict[str, dict[str, np.ndarray]]:
            Nested mapping of dataset split to absolute and relative errors.
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


def get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent,
    max_latent,
    n_latent_to_check=6,
    *,
    epochs=200,
    batch_size=128,
    patience=20,
    hidden_units=(256, 128),
    activation='relu',
    learning_rate=1e-3,
    verbose=0,
):
    """Train autoencoders for several latent sizes and return error metrics.

    Args:
        y_all_scaled (np.ndarray): Scaled full dataset.
        y_train_scaled (np.ndarray): Scaled training dataset.
        y_test_scaled (np.ndarray): Scaled test dataset.
        data (Dataset): Dataset wrapper providing reference arrays.
        y_scaler (Scaler): Fitted scaler used to invert the scaling.
        min_latent (int): Smallest latent dimension to evaluate.
        max_latent (int): Largest latent dimension to evaluate.
        n_latent_to_check (int): Number of latent sizes to sample.
        epochs (int): Maximum training epochs per autoencoder.
        batch_size (int): Minibatch size for training and inference.
        patience (int): Early stopping patience in epochs.
        hidden_units (tuple[int, ...]): Dense layer widths for the network.
        activation (str): Activation applied to hidden layers.
        learning_rate (float): Adam learning rate.
        verbose (int): Verbosity passed to ``train_autoencoder``.

    Returns:
        tuple[
            dict[str, dict[str, dict[str, np.ndarray]]],
            dict[int, dict[str, list[float]]],
        ]:
            Reconstruction differences and the recorded loss histories.
    """

    latent_dims = np.linspace(
        min_latent,
        max_latent,
        num=n_latent_to_check,
        dtype=int,
    )
    latent_dims = np.unique(np.clip(latent_dims, 1, y_train_scaled.shape[1]))

    diffs = {'x': latent_dims}
    histories = {}
    for dataset in ['all', 'train', 'test']:
        diffs[dataset] = {}
        for type_diff in ['rel', 'abs']:
            diffs[dataset][type_diff] = {}
            for mean_or_max in ['mean', 'max']:
                diffs[dataset][type_diff][mean_or_max] = np.zeros(
                    len(latent_dims),
                )

    for idx, latent_dim in enumerate(latent_dims):
        train_start = perf_counter()
        autoencoder, history = train_autoencoder(
            y_train_scaled,
            y_test_scaled,
            latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            hidden_units=hidden_units,
            activation=activation,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        train_elapsed = perf_counter() - train_start
        print(
            f'----> Latent {latent_dim}: training took {train_elapsed:.2f}s'
        )

        inference_start = perf_counter()
        recon_all = autoencoder.predict(
            y_all_scaled,
            batch_size=batch_size,
            verbose=0,
        )
        recon_train = autoencoder.predict(
            y_train_scaled,
            batch_size=batch_size,
            verbose=0,
        )
        recon_test = autoencoder.predict(
            y_test_scaled,
            batch_size=batch_size,
            verbose=0,
        )
        inference_elapsed = perf_counter() - inference_start
        print(
            '----> Latent %s: encode/decode took %.2fs'
            % (latent_dim, inference_elapsed),
        )

        y_all_rec, y_train_rec, y_test_rec = inv_scale(
            recon_all,
            recon_train,
            recon_test,
            y_scaler,
        )

        diffs_tmp = diff(
            y_all_rec,
            y_train_rec,
            y_test_rec,
            data.y,
            data.y_train,
            data.y_test,
        )

        for dataset in ['all', 'train', 'test']:
            for type_diff in ['rel', 'abs']:
                diffs[dataset][type_diff]['mean'][idx] = np.mean(
                    diffs_tmp[dataset][type_diff],
                )
                diffs[dataset][type_diff]['max'][idx] = np.max(
                    diffs_tmp[dataset][type_diff],
                )

        histories[int(latent_dim)] = history
        print(
            f'Done latent dimension {latent_dim} '
            f'({idx + 1}/{len(latent_dims)})'
        )

    return diffs, histories


def plot(diffs, histories=None, *, base_name, title):
    """Save reconstruction diff plots and optional training histories.

    Args:
        diffs (dict[str, dict[str, dict[str, np.ndarray]]]):
            Error metrics to visualise.
        histories (dict[int, dict[str, list[float]]] | None):
            Optional training histories keyed by latent dimension.
        base_name (str): Prefix for the saved figure filenames.
        title (str): Title displayed on the reconstruction error plots.

    Returns:
        tuple[Path, Path | None]:
            Paths to the generated diff and history figures.
    """

    diff_path = OUTPUT_DIR / f'{base_name}_diffs.pdf'
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)

    for ndataset, dataset in enumerate(['all', 'train', 'test']):
        axs[0, ndataset].set_title(f'{dataset} - rel_diff')
        axs[1, ndataset].set_title(f'{dataset} - abs_diff')

        axs[0, ndataset].set_yscale('log')
        axs[1, ndataset].set_yscale('log')
        axs[0, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)
        axs[1, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)

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

    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(diff_path, dpi=150)
    plt.close(fig)

    hist_path = None
    if histories:
        hist_path = OUTPUT_DIR / f'{base_name}_histories.pdf'
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        for latent_dim, history in sorted(histories.items()):
            ax_hist.plot(history['loss'], label=f'{latent_dim} train')
            if 'val_loss' in history:
                ax_hist.plot(
                    history['val_loss'],
                    linestyle='--',
                    label=f'{latent_dim} val',
                )
        ax_hist.set_xlabel('epoch')
        ax_hist.set_ylabel('loss (MSE)')
        ax_hist.set_yscale('log')
        ax_hist.set_title('Autoencoder training histories')
        ax_hist.legend(ncol=2)
        fig_hist.tight_layout()
        fig_hist.savefig(hist_path, dpi=150)
        plt.close(fig_hist)

    return diff_path, hist_path


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    for config in SPECTRUM_CONFIGS:
        base_name = config['spectrum']
        diff_path = OUTPUT_DIR / f'{base_name}_diffs.pdf'
        hist_path = OUTPUT_DIR / f'{base_name}_histories.pdf'
        if diff_path.exists():
            print(
                f'----> Skipping {base_name}; output already present at '
                f'{diff_path}'
            )
            continue

        dataset_list = [
            Dataset().load(
                path=os.path.join(
                    ROOT,
                    f'{MODEL}/sample/{config["spectrum_type"]}_100_{dr}.fits',
                ),
                name=base_name,
                verbose=False,
            )
            for dr in DATASET_RANGES
        ]
        data = Dataset.join(dataset_list, verbose=True)

        data.train_test_split(0.9, 1543, verbose=True)

        y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
            data.y,
            data.y_train,
            data.y_test,
        )

        max_latent = y_train_scaled.shape[1]
        min_latent_eval = max(1, min(config['min_latent'], max_latent))
        zoom_min_latent = max(
            min_latent_eval,
            int(round(config['zoom_min_latent_fraction'] * max_latent)),
        )
        if zoom_min_latent >= max_latent:
            zoom_min_latent = max(min_latent_eval, max_latent - 1)
        print(
            f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}'
        )

        diffs, histories = get_latent_to_check(
            y_all_scaled,
            y_train_scaled,
            y_test_scaled,
            data,
            y_scaler,
            min_latent_eval,
            max_latent,
            n_latent_to_check=config['n_latent_to_check'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            patience=config['patience'],
            hidden_units=config['hidden_units'],
            activation=config['activation'],
            learning_rate=config['learning_rate'],
            verbose=config['verbose'],
        )

        diff_path, hist_path = plot(
            diffs,
            histories,
            base_name=base_name,
            title=f'{base_name} autoencoder reconstruction errors',
        )
        print(f'----> Saved diffs to {diff_path}')
        if hist_path is not None:
            print(f'----> Saved histories to {hist_path}')
