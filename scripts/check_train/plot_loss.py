"""Plot loss and validation loss per epoch for multiple training runs."""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss(roots, save_dir='.'):
    for root in roots:
        path = os.path.join(root, 'history_log.csv')
        data = np.genfromtxt(path, delimiter=',', names=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data['epoch'], data['loss'], label='loss')
        ax.plot(data['epoch'], data['val_loss'], label='val_loss')

        for idx in range(1, len(data['learning_rate'])):
            if data['learning_rate'][idx] != data['learning_rate'][idx - 1]:
                ax.axvline(x=data['epoch'][idx], color='k', linestyle='--',
                           label='LR change' if idx == 1 else None)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_title(os.path.basename(root))
        ax.legend()
        plt.tight_layout()

        if os.path.split(root)[-1] == '':
            fname = 'loss_{}.png'.format(os.path.basename(os.path.dirname(root)))
        else:
            fname = 'loss_{}.png'.format(os.path.basename(root))
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss per epoch for training runs.')
    parser.add_argument(
        '--roots',
        '-r',
        type=str,
        nargs='+',
        help='Paths to training run folders containing history_log.csv.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/ceph/hpc/home/bellinie/emu_like/output',
        help='Directory to save figures. Defaults to script directory.')
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Find all folders containing history_log.csv in the provided roots
    roots = []
    for root in args.roots:
        for folder, folders, files in os.walk(root):
            if 'history_log.csv' in files:
                roots.append(folder)

    plot_loss(roots, save_dir=save_dir)
    print(f"\nFigures saved to {save_dir}")
