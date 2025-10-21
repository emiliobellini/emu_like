#!/usr/bin/env python3

# # Test PCA

# ## Init

root = '/data/emilio/emu_like'
model = 'lcdm'
dataset_ranges = ['thin', 'std', 'ext']

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from emu_like.pca import PCA
from emu_like.datasets import Dataset
from emu_like.scalers import Scaler

OUTPUT_DIR = Path("/home/embellin/emu_like/output/test_pca")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def scale(y, y_train, y_test):
    # Scale
    y_scaler = Scaler.choose_one('MinMaxScaler')
    y_scaler.fit(y_train)

    y_all_scaled = y_scaler.transform(y)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    print('----> Done scale!')

    return y_all_scaled, y_train_scaled, y_test_scaled, y_scaler

def pca(n_components, y, y_train, y_test):
    # PCA
    y_pca = PCA(n_components=n_components)
    y_pca.fit(y_train)

    y_all_pca = y_pca.transform(y)
    y_train_pca = y_pca.transform(y_train)
    y_test_pca = y_pca.transform(y_test)
    print('----> Done PCA!')
    return y_all_pca, y_train_pca, y_test_pca, y_pca

def inv_pca(y_all_pca, y_train_pca, y_test_pca, y_pca):
    # Inverse PCA
    y_all = y_pca.inverse_transform(y_all_pca)
    y_train = y_pca.inverse_transform(y_train_pca)
    y_test = y_pca.inverse_transform(y_test_pca)
    print('----> Done inverse PCA!')
    return y_all, y_train, y_test

def inv_scale(y_all_scaled, y_train_scaled, y_test_scaled, y_scaler):
    # Inverse Scale
    y_all = y_scaler.inverse_transform(y_all_scaled)
    y_train = y_scaler.inverse_transform(y_train_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)
    print('----> Done inverse scale!')
    return y_all, y_train, y_test

def select(y_all, y_train, y_test, pca_modes):
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
    diffs = {
        'all': {
            'rel': np.mean(np.abs(y_all/y_all_ref -1.), axis=0),
            'abs': np.mean(np.abs(y_all - y_all_ref), axis=0),
        },
        'train': {
            'rel': np.mean(np.abs(y_train/y_train_ref -1.), axis=0),
            'abs': np.mean(np.abs(y_train - y_train_ref), axis=0),
        },
        'test': {
            'rel': np.mean(np.abs(y_test/y_test_ref -1.), axis=0),
            'abs': np.mean(np.abs(y_test - y_test_ref), axis=0),
        },
    }
    print('----> Done diff!')
    return diffs

def get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, min_mode, max_mode, n_modes_to_check=20):

    modes_to_check = np.linspace(min_mode, max_mode, num=n_modes_to_check, dtype=int)

    # Init arrays
    diffs = {
        'x': modes_to_check,
    }
    for dataset in ['all', 'train', 'test']:
        diffs[dataset] = {}
        for type_diff in ['rel', 'abs']:
            diffs[dataset][type_diff] = {}
            for mean_or_max in ['mean', 'max']:
                diffs[dataset][type_diff][mean_or_max] = np.zeros(len(modes_to_check))

    for nmode, mode in enumerate(modes_to_check):
        y_all_tmp, y_train_tmp, y_test_tmp = select(y_all, y_train, y_test, range(mode))

        y_all_tmp, y_train_tmp, y_test_tmp = inv_pca(y_all_tmp, y_train_tmp, y_test_tmp, y_pca)
        y_all_tmp, y_train_tmp, y_test_tmp = inv_scale(y_all_tmp, y_train_tmp, y_test_tmp, y_scaler)

        diffs_tmp = diff(y_all_tmp, y_train_tmp, y_test_tmp, data.y, data.y_train, data.y_test)

        for dataset in ['all', 'train', 'test']:
            for type_diff in ['rel', 'abs']:
                diffs[dataset][type_diff]['mean'][nmode] = np.mean(diffs_tmp[dataset][type_diff])
                diffs[dataset][type_diff]['max'][nmode] = np.max(diffs_tmp[dataset][type_diff])

        print('Done mode {} ({}/{})'.format(mode, nmode+1, n_modes_to_check))

    return diffs

def plot(diffs, y_pca, output_path, title):
    output_path = Path(output_path)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)

    for ndataset, dataset in enumerate(['all', 'train', 'test']):
        axs[0, ndataset].set_title('{} - rel_diff'.format(dataset))
        axs[1, ndataset].set_title('{} - abs_diff'.format(dataset))

        axs[0, ndataset].set_yscale('log')
        axs[1, ndataset].set_yscale('log')
        axs[0, ndataset].set_xlim(diffs['x'][0]-1, diffs['x'][-1]+1)
        axs[1, ndataset].set_xlim(diffs['x'][0]-1, diffs['x'][-1]+1)

        singular_values = y_pca.pca.singular_values_/y_pca.pca.singular_values_[0]

        # First row, relative difference
        axs[0, ndataset].plot(diffs['x'], diffs[dataset]['rel']['mean'], label='mean')
        axs[0, ndataset].plot(diffs['x'], diffs[dataset]['rel']['max'], label='max')
        axs[0, ndataset].plot(singular_values * diffs[dataset]['rel']['mean'][0], label='singular values norm')

        # Second row, absolute difference
        axs[1, ndataset].plot(diffs['x'], diffs[dataset]['abs']['mean'], label='mean')
        axs[1, ndataset].plot(diffs['x'], diffs[dataset]['abs']['max'], label='max')
        axs[1, ndataset].plot(singular_values * diffs[dataset]['abs']['mean'][0], label='singular values norm')

    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

if False:
    # ## Pk matter

    spectrum = 'pk_m'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## Pk cb

    spectrum = 'pk_cb'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## Pk Weyl

    spectrum = 'pk_weyl'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## fk matter

    spectrum = 'fk_m'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## fk cb

    spectrum = 'fk_cb'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## fk Weyl

    spectrum = 'fk_weyl'
    spectrum_type = 'pk'
    n_modes = 600

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

    # ## Cl TT

    spectrum = 'cl_TT_lensed'
    spectrum_type = 'cl'
    n_modes = 2999

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

    y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
    y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

    diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

    plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

# ## Cl TE

spectrum = 'cl_TE_lensed'
spectrum_type = 'cl'
n_modes = 2999

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

y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

# ## Cl EE

spectrum = 'cl_EE_lensed'
spectrum_type = 'cl'
n_modes = 2999

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

y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

# ## Cl BB

spectrum = 'cl_BB_lensed'
spectrum_type = 'cl'
n_modes = 2999

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

y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

# ## Cl pp

spectrum = 'cl_pp_lensed'
spectrum_type = 'cl'
n_modes = 2999

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

y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

# ## Cl Tp

spectrum = 'cl_Tp_lensed'
spectrum_type = 'cl'
n_modes = 2999

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

y_all, y_train, y_test, y_scaler = scale(data.y, data.y_train, data.y_test)
y_all, y_train, y_test, y_pca = pca(n_modes, y_all, y_train, y_test)

diffs = get_modes_to_check(y_all, y_train, y_test, data, y_scaler, y_pca, 1, n_modes, n_modes_to_check=20)

plot(diffs, y_pca, OUTPUT_DIR / f"{spectrum}_pca_errors.pdf", f"{spectrum} PCA reconstruction errors")

