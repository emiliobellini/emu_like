#!/usr/bin/env python3

# # Test Autoencoder

# ## Init

root = '/data/emilio/emu_like'
model = 'lcdm'
dataset_ranges = ['thin', 'std', 'ext']

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from emu_like.datasets import Dataset
from emu_like.scalers import Scaler

# Ensure deterministic behaviour as much as possible
np.random.seed(0)
tf.random.set_seed(0)

OUTPUT_DIR = Path("/home/embellin/emu_like/output/test_autoencoder_lcdm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def scale(y, y_train, y_test, scaler_name='MinMaxCommonScaler'):
    """Scale arrays using one of the project scalers."""
    y_scaler = Scaler.choose_one(scaler_name)
    y_scaler.fit(y_train)

    y_all_scaled = y_scaler.transform(y)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    print('----> Done scale!')

    return y_all_scaled, y_train_scaled, y_test_scaled, y_scaler


def build_autoencoder(input_dim, latent_dim, hidden_units=(256, 128), activation='relu'):
    """Create a symmetric fully-connected autoencoder."""
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
    latent = layers.Dense(latent_dim, name='latent')(x)

    x = latent
    for units in reversed(hidden_units):
        x = layers.Dense(units, activation=activation)(x)
    outputs = layers.Dense(input_dim)(x)

    autoencoder = keras.Model(inputs, outputs, name=f'autoencoder_{latent_dim}')
    return autoencoder


def train_autoencoder(y_train, y_val, latent_dim, *, epochs=200, batch_size=128,
                      patience=20, hidden_units=(256, 128), activation='relu',
                      learning_rate=1e-3, verbose=0):
    """Train an autoencoder and return the fitted model and history."""
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32) if y_val is not None else None

    autoencoder = build_autoencoder(y_train.shape[1], latent_dim,
                                    hidden_units=hidden_units,
                                    activation=activation)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    callbacks = []
    if y_val is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=max(1, patience // 2), verbose=verbose))
    else:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='loss', patience=patience, restore_best_weights=True))

    history = autoencoder.fit(
        y_train, y_train,
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
    """Inverse scaling helper."""
    y_all = y_scaler.inverse_transform(y_all_scaled)
    y_train = y_scaler.inverse_transform(y_train_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)
    print('----> Done inverse scale!')
    return y_all, y_train, y_test


def diff(y_all, y_train, y_test, y_all_ref, y_train_ref, y_test_ref):
    """Compute mean and max absolute/relative differences."""
    diffs = {
        'all': {
            'rel': np.mean(np.abs(y_all / y_all_ref - 1.), axis=0),
            'abs': np.mean(np.abs(y_all - y_all_ref), axis=0),
        },
        'train': {
            'rel': np.mean(np.abs(y_train / y_train_ref - 1.), axis=0),
            'abs': np.mean(np.abs(y_train - y_train_ref), axis=0),
        },
        'test': {
            'rel': np.mean(np.abs(y_test / y_test_ref - 1.), axis=0),
            'abs': np.mean(np.abs(y_test - y_test_ref), axis=0),
        },
    }
    print('----> Done diff!')
    return diffs

def get_latent_to_check(y_all_scaled, y_train_scaled, y_test_scaled, data, y_scaler,
                        min_latent, max_latent, n_latent_to_check=6, *,
                        epochs=200, batch_size=128, patience=20,
                        hidden_units=(256, 128), activation='relu', learning_rate=1e-3,
                        verbose=0):
    """Train autoencoders with different bottleneck sizes and collect diffs."""
    latent_dims = np.linspace(min_latent, max_latent, num=n_latent_to_check, dtype=int)
    latent_dims = np.unique(np.clip(latent_dims, 1, y_train_scaled.shape[1]))

    diffs = {'x': latent_dims}
    histories = {}
    for dataset in ['all', 'train', 'test']:
        diffs[dataset] = {}
        for type_diff in ['rel', 'abs']:
            diffs[dataset][type_diff] = {}
            for mean_or_max in ['mean', 'max']:
                diffs[dataset][type_diff][mean_or_max] = np.zeros(len(latent_dims))

    for idx, latent_dim in enumerate(latent_dims):
        autoencoder, history = train_autoencoder(
            y_train_scaled, y_test_scaled,
            latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            hidden_units=hidden_units,
            activation=activation,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        recon_all = autoencoder.predict(y_all_scaled, batch_size=batch_size, verbose=0)
        recon_train = autoencoder.predict(y_train_scaled, batch_size=batch_size, verbose=0)
        recon_test = autoencoder.predict(y_test_scaled, batch_size=batch_size, verbose=0)

        y_all_rec, y_train_rec, y_test_rec = inv_scale(
            recon_all, recon_train, recon_test, y_scaler)

        diffs_tmp = diff(y_all_rec, y_train_rec, y_test_rec,
                         data.y, data.y_train, data.y_test)

        for dataset in ['all', 'train', 'test']:
            for type_diff in ['rel', 'abs']:
                diffs[dataset][type_diff]['mean'][idx] = np.mean(diffs_tmp[dataset][type_diff])
                diffs[dataset][type_diff]['max'][idx] = np.max(diffs_tmp[dataset][type_diff])

        histories[int(latent_dim)] = history
        print(f'Done latent dimension {latent_dim} ({idx + 1}/{len(latent_dims)})')

    return diffs, histories

def plot(diffs, histories=None, *, base_name, title):
    """Save reconstruction diff plots (and optional training histories)."""
    diff_path = OUTPUT_DIR / f"{base_name}_diffs.pdf"
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)

    for ndataset, dataset in enumerate(['all', 'train', 'test']):
        axs[0, ndataset].set_title(f'{dataset} - rel_diff')
        axs[1, ndataset].set_title(f'{dataset} - abs_diff')

        axs[0, ndataset].set_yscale('log')
        axs[1, ndataset].set_yscale('log')
        axs[0, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)
        axs[1, ndataset].set_xlim(diffs['x'][0] - 1, diffs['x'][-1] + 1)

        axs[0, ndataset].plot(diffs['x'], diffs[dataset]['rel']['mean'], label='mean')
        axs[0, ndataset].plot(diffs['x'], diffs[dataset]['rel']['max'], label='max')

        axs[1, ndataset].plot(diffs['x'], diffs[dataset]['abs']['mean'], label='mean')
        axs[1, ndataset].plot(diffs['x'], diffs[dataset]['abs']['max'], label='max')

    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(diff_path, dpi=150)
    plt.close(fig)

    if histories:
        hist_path = OUTPUT_DIR / f"{base_name}_histories.pdf"
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        for latent_dim, history in sorted(histories.items()):
            ax_hist.plot(history['loss'], label=f'{latent_dim} train')
            if 'val_loss' in history:
                ax_hist.plot(history['val_loss'], linestyle='--', label=f'{latent_dim} val')
        ax_hist.set_xlabel('epoch')
        ax_hist.set_ylabel('loss (MSE)')
        ax_hist.set_yscale('log')
        ax_hist.set_title('Autoencoder training histories')
        ax_hist.legend(ncol=2)
        fig_hist.tight_layout()
        fig_hist.savefig(hist_path, dpi=150)
        plt.close(fig_hist)
    else:
        hist_path = None

    return diff_path, hist_path

# ## Pk matter

spectrum = 'pk_m'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=1,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")
exit()

# ## Pk cb

spectrum = 'pk_cb'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Pk Weyl

spectrum = 'pk_weyl'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## fk matter

spectrum = 'fk_m'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## fk cb

spectrum = 'fk_cb'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## fk Weyl

spectrum = 'fk_weyl'
spectrum_type = 'pk'
min_latent = 16
n_latent_to_check = 8
zoom_min_latent_fraction = 0.75
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl TT

spectrum = 'cl_tt_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl TE

spectrum = 'cl_te_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl EE

spectrum = 'cl_ee_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl BB

spectrum = 'cl_bb_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl pp

spectrum = 'cl_pp_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

# ## Cl Tp

spectrum = 'cl_tp_lensed'
spectrum_type = 'cl'
min_latent = 32
n_latent_to_check = 7
zoom_min_latent_fraction = 0.85
zoom_n_latent_to_check = 5
autoencoder_epochs = 200
autoencoder_batch_size = 128
autoencoder_patience = 20
autoencoder_learning_rate = 1e-3
hidden_units = (512, 256, 128)
activation = 'relu'

data = [Dataset().load(
    path=os.path.join(root, '{}/sample/{}_100_{}.fits'.format(model, spectrum_type, dr)),
    name=spectrum,
    verbose=False)
    for dr in dataset_ranges]
data = Dataset.join(data, verbose=True)

data.train_test_split(
    0.9,
    1543,
    verbose=True)

y_all_scaled, y_train_scaled, y_test_scaled, y_scaler = scale(
    data.y, data.y_train, data.y_test)
max_latent = y_train_scaled.shape[1]
min_latent_eval = max(1, min(min_latent, max_latent))
zoom_min_latent = max(min_latent_eval, int(round(zoom_min_latent_fraction * max_latent)))
if zoom_min_latent >= max_latent:
    zoom_min_latent = max(min_latent_eval, max_latent - 1)
print(f'max_latent = {max_latent}, zoom_min_latent = {zoom_min_latent}')

diffs, histories = get_latent_to_check(
    y_all_scaled,
    y_train_scaled,
    y_test_scaled,
    data,
    y_scaler,
    min_latent_eval,
    max_latent,
    n_latent_to_check=n_latent_to_check,
    epochs=autoencoder_epochs,
    batch_size=autoencoder_batch_size,
    patience=autoencoder_patience,
    hidden_units=hidden_units,
    activation=activation,
    learning_rate=autoencoder_learning_rate,
    verbose=0,
)

plot(diffs, histories, base_name=spectrum, title=f"{spectrum} autoencoder reconstruction errors")

