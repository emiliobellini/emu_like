"""Plot loss and validation loss per epoch for multiple training runs."""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
matplotlib.use('Agg')


def last_relative_improvement(
        values,
        min_rel_delta=1e-3,
        min_abs_delta=1.-14):

    best = None
    best_idx = None

    for idx, current in enumerate(values):
        if not np.isfinite(current):
            continue

        if best is None:
            best = current
            best_idx = idx
            continue

        required_improvement = max(
            min_abs_delta,
            min_rel_delta * abs(best),
        )

        if best - current > required_improvement:
            best = current
            best_idx = idx

    return best_idx


def plot_loss(roots, save_dir=None):
    for root in roots:
        path = os.path.join(root, 'history_log.csv')
        data = np.genfromtxt(path, delimiter=',', names=True)

        # Fix save directory
        if save_dir is None:
            save_path = os.path.join(root, 'loss_vs_epoch.png')
        else:
            if os.path.split(root)[-1] == '':
                fname = 'loss_vs_epoch_{}.png'.format(
                    os.path.basename(os.path.dirname(root)))
            else:
                fname = 'loss_vs_epoch_{}.png'.format(os.path.basename(root))
            save_path = os.path.join(save_dir, fname)

        # Last epoch
        last_epoch = data['epoch'][-1]

        # Absolute Best validation loss
        abs_best_idx = np.argmin(data['val_loss'])
        abs_best_epoch = data['epoch'][abs_best_idx]
        abs_best_val_loss = data['val_loss'][abs_best_idx]
        abs_best_loss = data['loss'][abs_best_idx]

        # Relative Best validation loss
        rel_best_idx = last_relative_improvement(
            data['val_loss'],
            min_rel_delta=1e-3,
            min_abs_delta=0.0)
        rel_best_epoch = data['epoch'][rel_best_idx]
        rel_best_val_loss = data['val_loss'][rel_best_idx]
        rel_best_loss = data['loss'][rel_best_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data['epoch'], data['val_loss'], label='val_loss')
        ax.plot(data['epoch'], data['loss'], label='loss')

        try:
            ax.plot(
                data['epoch'],
                data['val_loss_pk'],
                '--',
                label='val_loss_pk')
            ax.plot(
                data['epoch'],
                data['loss_pk'],
                '--',
                label='loss_pk')
            ax.plot(data['epoch'], data['val_loss_fk'], label='val_loss_fk')
            ax.plot(data['epoch'], data['loss_fk'], label='loss_fk')
        except ValueError:
            pass

        # Plot abs best epoch
        ax.plot(
            abs_best_epoch,
            abs_best_val_loss,
            'ro',
            markersize=4,
            label='Absolute Best val_loss: {:.2e}'.format(abs_best_val_loss))
        ax.plot(
            abs_best_epoch,
            abs_best_loss,
            'go',
            markersize=4,
            label='Loss at absolute best val_loss: {:.2e}'.format(
                abs_best_loss))

        # Plot rel best epoch
        ax.plot(
            rel_best_epoch,
            rel_best_val_loss,
            'bo',
            markersize=4,
            label='Relative Best val_loss: {:.2e}'.format(rel_best_val_loss))
        ax.plot(
            rel_best_epoch,
            rel_best_loss,
            'mo',
            markersize=4,
            label='Loss at relative best val_loss: {:.2e}'.format(
                rel_best_loss))

        for idx in range(1, len(data['learning_rate'])):
            if data['learning_rate'][idx] != data['learning_rate'][idx - 1]:
                ax.axvline(x=data['epoch'][idx], color='k', linestyle='--',
                           label='LR change' if idx == 1 else None)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(
            '{}. Epochs:\n'
            ' Last: {} | Abs Best: {} (diff: {}) | Rel Best: {} (diff: {})'
            ''.format(
                os.path.basename(root),
                int(last_epoch),
                int(abs_best_epoch),
                int(last_epoch - abs_best_epoch),
                int(rel_best_epoch),
                int(last_epoch - rel_best_epoch)))
        ax.legend()
        plt.tight_layout()

        fig.savefig(save_path,
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot loss per epoch for training runs.')
    parser.add_argument(
        '--roots',
        '-r',
        type=str,
        nargs='+',
        help='Paths to training run folders containing history_log.csv.')
    parser.add_argument(
        '--save-dir',
        '-s',
        type=str,
        help='Directory to save figures. Defaults to script directory.')
    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # Find all folders containing history_log.csv in the provided roots
    roots = []
    for root in args.roots:
        for folder, folders, files in os.walk(root):
            if 'history_log.csv' in files:
                roots.append(folder)

    plot_loss(roots, save_dir=args.save_dir)
