"""Serial range workers and transactional, row-indexed FITS merging.

The numbered range FITS is itself the checkpoint. Locks are advisory: old
processes started before this implementation must stop before merging.
"""
from contextlib import contextmanager, ExitStack
from functools import wraps
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from astropy.io import fits
from . import io
from .x_samplers import XSampler
from .y_models import YModel

MASK = 'SAMPLE_DONE'
ROWS = 'ROW_IDS'


@contextmanager
def lock(path):
    """Fail immediately on a competing writer; kernel releases on job death."""
    path = Path(str(Path(path).resolve()) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another job is writing {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    # Keep the lock inode: deleting it could allow two independent locks.


def writer_lock(argument):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            path = inspect.signature(function).bind(
                *args, **kwargs).arguments.get(argument)
            if path is None:
                return function(*args, **kwargs)
            with lock(path):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def atomic_write(hdus, path):
    """
    Publish a complete, verified FITS, leaving the previous file on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            hdus.writeto(stream, checksum=True)
            stream.flush()
            os.fsync(stream.fileno())
        with fits.open(temporary, memmap=False) as check:
            check.verify('exception')
            if any(h.verify_checksum() != 1 or h.verify_datasum() != 1
                   for h in check):
                raise IOError('FITS checkpoint checksum verification failed')
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source(path):
    """
    Read immutable settings, inputs and reference grids; ignore growing y.
    """
    settings = io.FitsFile(str(path)).get_header(0, unflat_dict=True)
    sampler = XSampler.choose_one(
        settings['x_sampler']['name'], settings['params'],
        **(settings['x_sampler']['args'] or {}))
    spec = settings['y_model']
    with fits.open(path, memmap=False) as hdus:
        x = np.array(hdus[sampler.x_key].data, copy=True)
    model = YModel.choose_one(
        spec['name'], settings['params'], spec['outputs'],
        len(x), **(spec['args'] or {}))
    model.load(fname=str(path))
    digest = hashlib.sha256(repr(settings).encode())
    with fits.open(path, memmap=False) as hdus:
        excluded = {key.upper() for key in model.y_keys} | {MASK}
        for hdu in hdus[1:]:
            if hdu.name.upper() not in excluded:
                digest.update(hdu.name.encode())
                digest.update(str(hdu.data.shape).encode())
                digest.update(hdu.data.tobytes())
    return x, model, digest.hexdigest()


def completion(hdus, keys, count):
    if MASK in hdus:
        mask = np.asarray(hdus[MASK].data)
        if mask.shape != (count,) or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid completion mask')
        if any(key not in hdus or len(hdus[key].data) != count
               for key in keys):
            raise ValueError(
                'Masked outputs must have the full input row count')
        return mask.astype(bool)
    lengths = [len(hdus[key].data) if key in hdus else 0 for key in keys]
    if len(set(lengths)) != 1 or lengths[0] > count:
        raise ValueError(
            'Legacy outputs have inconsistent row counts; '
            'repair before merging')
    return np.arange(count) < lengths[0]


def read_part(path, identity, keys, widths, count):
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].header.get('SRCID') != identity:
            raise ValueError(f'{path}: input/settings/reference mismatch')
        for hdu in hdus:
            if 'CHECKSUM' in hdu.header and (
                    hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1):
                raise ValueError(f'{path}: damaged checkpoint checksum')
        start, stop = hdus[0].header['START'], hdus[0].header['STOP']
        rows = np.array(hdus[ROWS].data, copy=True)
        if (not (0 <= start < stop <= count)
                or rows.ndim != 1 or rows.dtype.kind not in 'iu'):
            raise ValueError(f'{path}: invalid row range')
        if (len(np.unique(rows)) != len(rows)
                or np.any((rows < start) | (rows >= stop))):
            raise ValueError(f'{path}: invalid or duplicate row IDs')
        values = [np.array(hdus[key].data, copy=True) for key in keys]
        if any(value.shape != (len(rows), width)
               for value, width in zip(values, widths)):
            raise ValueError(f'{path}: invalid output shapes')
        return start, stop, rows, values


def sample_range(path, start, stop, save_interval=100, timeout=None,
                 pending_only=False):
    """Compute [start, stop) serially. Rerunning resumes the numbered file."""
    path = Path(path).resolve()
    part = path.parent / (path.stem + f'.rows-{start:09d}-{stop:09d}.fits')
    if save_interval is None:
        save_interval = 100
    if not isinstance(save_interval, int) or save_interval < 1:
        raise ValueError('save_interval must be a positive integer')
    with lock(part):
        x, model, identity = source(path)
        if not 0 <= start < stop <= len(x):
            raise ValueError(f'Require 0 <= start < stop <= {len(x)}')
        keys, widths = model.y_keys, model.get_n_y()
        rows, values = [], [[] for _ in keys]
        if part.exists():
            old_start, old_stop, previous, old_values = read_part(
                part, identity, keys, widths, len(x))
            if (old_start, old_stop) != (start, stop):
                raise ValueError('Checkpoint range mismatch')
            rows = previous.tolist()
            values = [list(value) for value in old_values]
        done = set(rows)
        if pending_only:
            with fits.open(path, memmap=False) as hdus:
                done.update(np.flatnonzero(completion(hdus, keys, len(x))))

        def checkpoint():
            hdus = fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                {'SRCID': identity, 'START': start, 'STOP': stop})),
                fits.ImageHDU(np.asarray(rows, dtype=np.int64), name=ROWS)])
            for key, width, value in zip(keys, widths, values):
                hdus.append(fits.ImageHDU(
                    np.asarray(value, dtype=float).reshape(-1, width),
                    name=key))
            atomic_write(hdus, part)
        started = time.monotonic()
        unsaved = 0
        try:
            for idx in range(start, stop):
                if idx in done:
                    continue
                result = model.evaluate(x[idx], idx)
                result = [np.asarray(y).reshape(-1) for y in result]
                if len(result) != len(keys) or any(
                        y.shape != (w,) for y, w in zip(result, widths)):
                    raise ValueError('Model returned invalid output shapes')
                for value, y in zip(values, result):
                    value.append(y.copy())
                rows.append(idx)
                unsaved += 1
                if unsaved >= save_interval:
                    checkpoint()
                    unsaved = 0
                if (timeout is not None
                        and time.monotonic() - started >= timeout * 3600):
                    break
        finally:
            if unsaved or not part.exists():
                checkpoint()
    return str(part)


def merge_ranges(path, parts, cleanup=True, already_locked=False):
    """Merge in supplied order (last wins). Caller must stop legacy writers."""
    path = Path(path).resolve()
    parts = [Path(p).resolve() for p in parts]
    if not parts or len(set(parts)) != len(parts) or path in parts:
        raise ValueError(
            'Provide distinct range files, excluding the main FITS')
    with ExitStack() as stack:
        if not already_locked:
            stack.enter_context(lock(path))
        for part in sorted(parts):
            stack.enter_context(lock(part))
        x, model, identity = source(path)
        keys, widths = model.y_keys, model.get_n_y()
        with fits.open(path, memmap=False) as original:
            hdus = fits.HDUList([h.copy() for h in original])
        mask = completion(hdus, keys, len(x))
        arrays = []
        for key, width in zip(keys, widths):
            array = np.full((len(x), width), np.nan)
            if key in hdus:
                old = hdus[key].data
                if old.shape[1:] != (width,):
                    raise ValueError('Main output width mismatch')
                array[:len(old)] = old
            else:
                helper = io.FitsFile(str(path))
                header = helper._delistify(helper._flatten_dict(
                    model.get_y_headers()[len(arrays)]))
                hdus.append(fits.ImageHDU(
                    name=key, header=fits.Header(header)))
            arrays.append(array)
        for part in parts:
            _, _, rows, values = read_part(
                part, identity, keys, widths, len(x))
            for array, value in zip(arrays, values):
                array[rows] = value
            mask[rows] = True
        for key, array in zip(keys, arrays):
            hdus[key].data = array
        if MASK in hdus:
            hdus[MASK].data = mask.astype(np.uint8)
        else:
            hdus.append(fits.ImageHDU(mask.astype(np.uint8), name=MASK))
        atomic_write(hdus, path)
        if cleanup:
            for part in parts:
                part.unlink()
    return int(mask.sum())
