import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from emu_like.datasets import Dataset


def parser():
    parser = argparse.ArgumentParser()
    # Files to inspect
    parser.add_argument(
        '--files',
        '-f',
        type=str,
        nargs='+',
        help='Fits files to inspect.')
    # Output folder
    parser.add_argument(
        '--output-folder',
        '-o',
        type=str,
        default='output',
        help='Output folder to save plots.')
    # Name of the spectrum to inspect
    parser.add_argument(
        '--name',
        '-n',
        type=str,
        help='Name of the spectrum to inspect.')
    # Scalers
    parser.add_argument(
        '--rescale-x',
        '-rx',
        type=str,
        help='Scalers for x-axis.')
    parser.add_argument(
        '--rescale-y',
        '-ry',
        type=str,
        help='Scalers for y-axis.')
    # PCA components
    parser.add_argument(
        '--num-pca-x',
        '-nx',
        type=int,
        help='Number of PCA components for x-axis.')
    parser.add_argument(
        '--num-pca-y',
        '-ny',
        type=int,
        help='Number of PCA components for y-axis.')
    # Other options
    parser.add_argument(
        '--keep-non-finite',
        '-knf',
        action='store_true',
        help='Keep non-finite values in the dataset.')
    parser.add_argument(
        '--frac-train',
        '-ft',
        type=float,
        default=0.9,
        help='Fraction of data to use for training.')
    parser.add_argument(
        '--train-test-random-seed',
        '-s',
        type=int,
        default=1543,
        help='Random seed for train-test split.')
    parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')

    args = parser.parse_args()
    return args


def get_model_name(files):
    choice = files[0].split('/')
    # Index of sample
    try:
        idx = choice.index('sample')
    except ValueError:
        idx = None

    if idx is None:
        # Remove ..
        choice = [x for x in choice if x != '..']
        # Join
        model = '_'.join(choice)
    else:
        model = choice[idx - 1]

    return model


def get_data_ranges_name(files):
    choice = [file.split('/')[-1] for file in files]
    choice = [x.split('_')[-1] for x in choice]
    choice = [x.split('.')[0] for x in choice]
    choice = '_'.join(choice)
    return choice


def stack_train_test(data):
    """Stack training and testing data."""
    x = np.vstack((data.x_train, data.x_test))
    y = np.vstack((data.y_train, data.y_test))
    return x, y


def plot_mode_density(
        samples,
        mode_values=None,
        n_bins=100,
        n_levels=30,
        cmap='viridis',
        zero_color='lightgray',
        logy=False,
        logx=False,
        xlim=None,
        ylim=None,
        save_path=None):
    """
    Visualise the per-mode sample density of a 2D array with shape (n_s, n_k).

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_s, n_k); each column holds the n_s samples of one mode.
    mode_values : array-like or None
        Optional x-axis ticks for the modes. Defaults to simple indices.
    n_bins : int
        Number of bins used to estimate the 1D density along the sample axis.
    levels : int
        Number of contour levels for plt.contourf.
    cmap : str
        Matplotlib colormap name for the filled contours.
    """
    if samples.ndim != 2:
        raise ValueError('samples must be a 2D array (n_s, n_k)')

    n_s, n_k = samples.shape
    if mode_values is None:
        mode_values = np.arange(n_k)

    print(samples.min(), samples.max())
    y_edges = np.linspace(samples.min(), samples.max(), n_bins + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    density = np.empty((n_bins, n_k))
    for j in range(n_k):
        hist, _ = np.histogram(samples[:, j], bins=y_edges, density=True)
        density[:, j] = hist

    positive = density[density > 0]
    if positive.size == 0:
        raise ValueError('All densities are zero; adjust binning.')

    min_pos = positive.min()
    max_pos = positive.max()
    levels = np.linspace(min_pos, max_pos, n_levels)

    X, Y = np.meshgrid(mode_values, y_centers)
    if logx:
        X += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(X, Y, density, levels=levels, cmap=cmap, extend='min')
    contour.cmap.set_under(zero_color)     # color for density == 0
    contour.changed()

    ax.set_xlabel('Mode index' if mode_values is None else 'Mode')
    ax.set_ylabel('Sample value')
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Density')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    return


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Parse command line arguments
    args = parser()

    # Get model
    model_name = get_model_name(args.files)

    # Get data ranges
    data_ranges_name = get_data_ranges_name(args.files)

    # Output filename
    # fname = '{}_{}_dr_{}_rx_{}_ry_{}_nx_{}_ny_{}.png'.format(
    #     args.name,
    #     model_name,
    #     data_ranges_name,
    #     args.rescale_x,
    #     args.rescale_y,
    #     args.num_pca_x,
    #     args.num_pca_y)
    fname = '{}_{}_dr_{}_res_{}_pca_{}.png'.format(
        args.name,
        model_name,
        data_ranges_name,
        args.rescale_y,
        args.num_pca_y)

    # Preliminary checks
    if args.files is None:
        raise ValueError(
            'No fits files provided. Use --files to specify files.')
    if args.name is None:
        raise ValueError(
            'No spectrum name provided. Use --name to specify the spectrum.')


    # Load data
    data = [Dataset().load(
        path=file,
        name=args.name) for file in args.files]

    # Remove non finite "y"
    if not args.keep_non_finite:
        data = [d.remove_non_finite(verbose=False) for d in data]

    # Join all datasets
    data = Dataset.join(data, verbose=True)

    # Split training and testing samples
    data.train_test_split(
        args.frac_train,
        args.train_test_random_seed,
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    data.rescale(
        args.rescale_x,
        args.rescale_y,
        verbose=args.verbose)

    # If requested apply PCA on x and/or y
    data.apply_pca(
        args.num_pca_x,
        args.num_pca_y,
        verbose=args.verbose)

    # Stack training and testing data
    x, y = stack_train_test(data)

    # # Plot mode density
    # plot_mode_density(
    #     y,
    #     mode_values=None,
    #     n_bins=200,
    #     n_levels=30,
    #     cmap='viridis',
    #     zero_color='lightgray',
    #     logy=False,
    #     logx=True,
    #     # xlim=[0.9, None],
    #     # ylim=[-200.,200.],
    #     save_path=os.path.join(args.output_folder, fname))

    plt.plot(np.arange(y.shape[1])+1, y[:3].T)
    plt.xscale('log')
    plt.savefig('output/test.png')
