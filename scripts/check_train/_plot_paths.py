"""Include relative training paths in diagnostic filenames."""

import os
from pathlib import Path


def source_root(roots):
    """Choose a shared base, retaining a directly selected run's name."""
    paths = [Path(root).absolute() for root in roots]
    base = Path(os.path.commonpath(paths))
    if (base / 'history_log.csv').is_file():
        base = base.parent
    return base


def plot_path(root, save_dir, filename, relative_to=None):
    """Flatten the relative run path into filenames under save_dir."""
    root = Path(root).absolute()
    if save_dir is None:
        return root / filename
    if relative_to is None:
        # Direct Python calls can infer the conventional training root.
        relative_to = next((p for p in root.parents if p.name == 'train'),
                           root.parent)
    relative = root.relative_to(Path(relative_to).absolute())
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    filename = Path(filename)
    label = '_'.join(relative.parts)
    return destination / f'{filename.stem}_{label}{filename.suffix}'
