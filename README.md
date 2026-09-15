# emu_like

<!-- ## Description -->

`emu_like` is a library created to build and use emulators for cosmological likelihoods.

As described below, it allows to generate a sample of data, and use it to train an emulator with a user-specified architecture. In [Sampling functions](src/emu_like/sampling_functions.py) it is possible to find all the functions that can be sampled. On top of simple functions that can be used to familiarise with the code, it has an interface with the [Cobaya](https://cobaya.readthedocs.io/en/latest/) sampler, which allows to easily build an emulator for any of the likelihoods implemented in it.

The code can be used either as a Python module or as a standalone code. As a standalone code there are three main pipelines:
- [sample](pipelines/sample.py): to generate and store a sample of data that will be used to train the emulator;
- [train](pipelines/train.py): to train the emulator on sampled data and store it;
- [mcmc](pipelines/mcmc.py): this is used to run an mcmc with an emulator.

As a Python module the two main classes that should be called are:
- [Sample](src/emu_like/sample.py): to generate, load and save samples of data;
- [Emulator](src/emu_like/emu.py): to train, load and save emulators. While the Emulator class is the base class of emulators, it is often convenient to call directly the type of emulator needed, e.g. [FFNNEmu](src/emu_like/ffnn_emu.py).

Each emulator returns likelihood values (depending on the number of likelihoods we emulate) providing a list of cosmological parameters.

**NOTE**: this code has a modular structure and can be generalised in a relatively simple way. Adding new samplers, scalers, emulators, loss functions (or anthing else) to the code can be done by simply copying a similar object and making the changes needed.


## Installation

To install the code, download it or clone it from github with
```
git clone https://github.com/emiliobellini/emu_like.git
```
Then, install it from source with
```
cd emu_like/
python -m pip install .
```
This will take care of installing all the dependencies.

### Class and hi_class

Cosmological sampling uses `hiclassy.HiClass` for both standard CLASS and hi_class models. Install `hiclassy` in the same Python environment before sampling, for example with `pip install '.[sampling]'`. It is optional when training on existing datasets.

### Cobaya

If you plan to use Cobaya and its likelihoods, follow instructions [here](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html). In short, to install the basic cosmological modules and likelihoods use
```
cobaya-install cosmo -p /path/to/packages
```

### Planck likelihoods

For the Planck clik likelihoods use, e.g.
```
cobaya-install planck_2018_highl_plik.TTTEEE
```

### Virtual environment

PS: we recommend to use it with a virtual environment, running
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```


## Usage

For TensorBoard installation, GPU profiling on the cluster, and measured
Sobolev performance, see [Profiling notes](docs/profiling.md).

If you plan to create your own pipeline, follow examples in [this folder](examples). They explain how to get a sample, train and use an emulator in simple scenarios.

The sample pipeline can be run from the main folder with
```
python main.py sample params.yaml -v
```
Ready to use parameter files can also be found in the main directory
- `simple_sample.yaml`: to sample a straight line in 1D on a grid;
- `planck_sample.yaml`: to sample the Planck likelihood using Cobaya, `hiclassy` and standard cosmological parameters. The `emu_like.cobaya_hiclass.HiClassTheory` adapter selects `HiClass` through Cobaya's CLASS interface.

The train pipeline can be run from the main folder with
```
python main.py train params.yaml -v
```
Ready to use parameter files can also be found in the main directory
- `simple_train.yaml`: to train an emulator that fits a straight line in 1D;
- `planck_train.yaml`: to train an emulator for the Planck total likelihood.

Finally, it is possible to test the emulator created with
```
python main.py mcmc planck_mcmc.yaml -v
```
where `planck_mcmc.yaml` contains all the information needed by the mcmc sampler.

**NOTE**: all the parameter files mentioned here can be used as a guidance to understand the input parameters at each step and their usage.


## Credits
The sampling part of this code depends on [Cobaya](https://cobaya.readthedocs.io/en/latest/) and its likelihoods.
If you use Cobaya, please cite its pre-print, [arXiv:2005.05290](https://arxiv.org/abs/2005.05290), and its ASCL record, [ascl:1910.019](https://ascl.net/1910.019).
If you use any of the likelihoods implemented in Cobaya, make sure you cite the relevant papers.


## Contributing
For bugs and feature requests consider using the [issue tracker](https://github.com/emiliobellini/emu_like/issues).

## License
`emu_like` is released under the GPL-3 license (see [LICENSE](LICENSE)).

## Weyl power and growth conventions

Write `W = (phi + psi)/2` and `Q_W(k,z) = k**4 P_W(k,z)`.
HiClass returns `Q_W` in `1/Mpc`, with k in `1/Mpc`. It is the power
spectrum of `k**2 W`, not the dimensionless power per logarithmic interval.

The clients `emu_like` and `hi_fast` accept q in `h/Mpc` and retain the
historical numerical normalization `S_W(q,z) = h**3 Q_W(h*q,z)` for
compatibility with existing datasets. This is not matter power in
`(Mpc/h)**3`; new `emu_like` Weyl headers label it `h^3/Mpc`.
Ratio targets additionally divide by their stored reference spectrum.

For every species, `f = (1/2) d ln P / d ln a = -(1+z)/(2P) dP/dz`.
For Weyl, P denotes the rescaled Weyl power; its growth can be negative.
Both clients differentiate HiClass power with second-order differences and
`dz = 1e-3`. They use a forward stencil near z=0. `emu_like` also uses a
backward stencil at the native upper time boundary; `hi_fast` reserves
coverage for its upper stencil. The fixed h normalization cancels in f.

Weyl requires a HiClass build exposing `pk_weyl`, `pk_weyl_lin`,
`get_pk_weyl`, and `get_pk_weyl_lin`. Both clients request `wPk`
automatically. `emu_like` follows the configured nonlinear setting and
rejects nonlinear Weyl requests; `hi_fast` explicitly uses linear power.
The unchanged `get_Weyl_pk_and_k_and_z` remains a comparison accessor, not
the source of client-side extrapolation.

Below the native k_min, the leading-order prescription is
`Q_W(k,z) = (k/k_min)**n_s Q_W(k_min,z)`. It assumes a k-independent Weyl
source at leading order and a single adiabatic analytic primordial power
law with `alpha_s = beta_s = 0`. Other primordial setups remain usable
within the native grid but are rejected below k_min. There is no high-k
or redshift extrapolation, and k must be strictly positive. Agreement
between implementations does not by itself establish this asymptotic
approximation's accuracy for every modified-gravity model.

Regenerating datasets updates targets and reference tables; existing
trained emulator weights are not changed by these code updates.

Before regenerating datasets, install the updated client checkouts into the
same virtual environment used by the batch jobs, for example:

```bash
python -m pip install --no-deps -e /cephhome/bellinie/emu_like -e /cephhome/bellinie/hi_fast
```

Confirm `emu_like.spectra.__file__` and `hi_fast.spectra.__file__` resolve to
the intended checkouts. Source-tree tests using `PYTHONPATH` do not update
previously installed package copies. The HiClass build in that environment
must also expose all four Weyl accessors listed above.

## Sampling independent row ranges on a cluster

Several jobs can compute serial ranges of the **same saved input table**. Each
writes a numbered FITS in the main file's folder. Row indices are zero-based;
`--stop-row` is exclusive. Run from this checkout with `PYTHONPATH=src` if your
Python installation otherwise imports an older installed copy of emu_like.

For a new dataset, first save the inputs and reference spectra once:

```bash
PYTHONPATH=src python main.py sample spectra_sample.yaml --prepare-only
```

For an existing dataset, use its existing YAML with `output.path` pointing to
the main FITS; preparation is unnecessary. Launch these as separate cluster
jobs (one process per job):

```bash
PYTHONPATH=src python main.py sample spectra_sample.yaml --start-row 50000 --stop-row 60000
PYTHONPATH=src python main.py sample spectra_sample.yaml --start-row 60000 --stop-row 70000
```

These commands read the settings, inputs and reference data saved in the main
FITS. Only `output.path`, `output.save_interval` and `output.timeout` come from
the YAML. Range mode implies resumption; `--resume` is unnecessary. Rerunning
the identical command resumes its numbered checkpoint, skipping saved row IDs.
A range always evaluates its assigned rows, even if they already exist in the
main file. A calculated NaN is a completed result, not an unfinished row.

For a main file called `sample.fits`, the first worker saves
`sample.rows-000050000-000060000.fits`. This file **is the checkpoint** and stores
all outputs and explicit row IDs together. Checkpoints are atomically replaced
every `save_interval` rows (default 100), on timeout, and on handled exceptions.
A killed node can lose work since its last checkpoint, without shifting rows.
Do not launch two workers for the same range filename; use distinct ranges.
Overlapping ranges with different filenames are supported.

After the workers stop, merge their files, using their actual output paths:

```bash
PYTHONPATH=src python main.py sample spectra_sample.yaml --merge-ranges \
  output/spectra/sample.rows-000050000-000060000.fits \
  output/spectra/sample.rows-000060000-000070000.fits
```

Files merge in the listed order: **later files overwrite earlier overlapping
rows**, including NaNs. Merging validates the input/settings/reference identity,
row IDs and output dimensions. It preserves existing results outside those
rows. Uncomputed rows contain NaNs, and the `SAMPLE_DONE` FITS extension records
completion independently of the output values. Ordinary `--resume` on a merged
dataset computes missing rows serially and merges them safely; use range jobs
for concurrency. Once merged, training should exclude nonfinite rows as usual.

The merge writes and verifies a temporary main FITS, then atomically replaces
the main file. Only after success are the listed range files/checkpoints deleted.
Use `--keep-ranges` to retain them. Allow space for a second main FITS during
merging and memory for the full dataset. Merge can also consume a stopped,
partially completed range; only its saved row IDs are copied.

**Jobs already running may continue while new range jobs compute, but they must
finish or stop before any merge into their main FITS.** Older processes do not
honor the new writer locks. Avoid reading their main FITS during an in-place
save; if a range job fails during initial loading, retry once that save finishes.
Do not change the input table or reference data between range jobs. New-code
writers and merges use advisory locks and reject competing writes; the cluster
filesystem must support POSIX locks and atomic rename. Small `.lock` files are
retained intentionally (they are not checkpoints); the OS releases their locks
when a process exits or dies. An uncatchable kill during saving may also leave
an unused `*.tmp` file; it is never used as a checkpoint.

After the first merge, use the updated code for every subsequent writer:
legacy versions cannot interpret the completion mask.
