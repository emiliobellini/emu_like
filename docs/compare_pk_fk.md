# Comparing standard and Sobolev power/growth

## Per-model histograms

`plot_histograms.py` also plots growth histograms independently for every
`pk_*` model it discovers, using the same derivative and histogram helpers:

```bash
python scripts/check_train/plot_histograms.py \
  --roots /path/to/train --save-dir output/histograms
```

For each dataset range, `fk_*_<range>_accuracy_absolute_histograms.png` and
`fk_*_<range>_accuracy_relative_percent_histograms.png` compare power-derived
growth with the matching FITS growth data. Sobolev runs overlay `eval_fk`
accuracy and add `consistency_absolute` and `consistency_relative_percent`
histograms of `eval_fk - fk_from_pk`. Each figure shows bin errors, per-spectrum
RMS, and per-spectrum maximum error. Files under `--save-dir` include the
relative training-run path, as with the existing plots.

Only the histograms are added: no derivative checks, diagnostic JSON/NPZ
files, or error-versus-k plots from `compare_pk_fk.py` are run. The existing
summary table and worst-mode plots remain available. Growth predictions
and FITS targets are compared in physical units using their respective saved
references. Relative errors divide by physical FITS growth, excluding bins
with absolute growth at or below `--growth-floor` (default `1e-6`).

All per-run histograms, summary statistics, and worst-mode plots use only
validation rows. The split is reconstructed from each run's `params.yaml`:
apply the training finite-row filter (including growth for Sobolev), join
the listed datasets in order, and split once using `frac_train` and
`train_test_random_seed`. Only ranges listed in that run are evaluated.
The original FITS files must be unchanged since training; an unseeded split
cannot be reconstructed and raises an error. Growth diagnostics further
exclude validation rows with nonfinite growth targets. The matching
`FK_*`, `REF_FK_*`, `K_RANGE_FK_*`, and `Z_ARRAY` FITS extensions are required.
Use `--growth-batch-size` (default 128) to control derivative memory, or
`--skip-growth-histograms` to skip the added plots. Standalone `fk_*` runs
retain their existing accuracy plots: a consistency comparison against a
separate power model requires the three-model workflow below.

For Sobolev, `eval_fk` is itself derivative-based; its comparison with the
independent reconstruction also probes reference interpolation differences,
as explained below.

## Joint validation comparison

Run `scripts/check_train/compare_pk_fk.py` after the three training runs have
stopped writing checkpoints. It loads their best validation checkpoints.
The original training FITS files must still contain the same data used during
training. Run from the repository root with the training Python environment:

```bash
python scripts/check_train/compare_pk_fk.py \
  --standard-pk /ceph/hpc/data/s25r06-05-users/lcdm_k/train/pk_m \
  --standard-fk /ceph/hpc/data/s25r06-05-users/lcdm_k/train/fk_m \
  --sobolev /ceph/hpc/data/s25r06-05-users/lcdm_k/train/sobolev/pk_m \
  --output output/lcdm_k_pk_fk_comparison
```

The default is 6,000 randomly selected common validation inputs, evaluated in
batches of 128. `--max-samples 0` evaluates the entire intersection. The script
reconstructs each model's seeded validation split **after its own finite-row
filter**, then intersects the original row IDs. The three models must use the
same ordered training FITS files, species, parameter ordering and k grid.
These are validation diagnostics, not a new blind test set.

Use `--k-min 0.001 --k-max 1` to restrict the plotted and summarized k range.
Use a new output directory for each comparison; existing results are not
silently overwritten. Full-grid predictions are retained even with a k cut.

## Outputs

For all selected rows together and separately for each input FITS file:

- `*_pk_relative_percent.png`: standard and Sobolev power errors against
  HiClass, in percent.
- `*_fk_relative_percent.png`: direct standard growth, growth derived from
  standard power, and Sobolev growth, against HiClass, in percent.
- `*_fk_absolute.png`: the same growth comparison in absolute growth units.
- `*_consistency_absolute.png`: direct minus power-derived growth for each
  method, in absolute growth units.

Each figure has distributions of absolute bin errors, RMS error per spectrum,
and maximum absolute error per spectrum. The x axes are logarithmic; exact
zero errors are omitted from the plots, but included in summary statistics.
Histogram densities describe distributions, not counts above a threshold.

`summary.json` contains checkpoint epochs/losses, input files, selected k range,
row counts, signed bias, RMS, absolute-error quantiles, consistency statistics,
and finite-difference checks. Relative growth residuals divide by HiClass growth;
bins with `abs(f_HiClass) <= --growth-floor` (default `1e-6`) are excluded and
counted as nonfinite. Per-spectrum relative statistics include only spectra
with every selected bin defined. Absolute growth statistics retain those bins.

`predictions.npz` stores physical predictions and HiClass targets, inputs,
full k grid, source-file indices and zero-based original row indices. It
includes `standard_fk_from_pk`, `sobolev_fk_from_pk`, `standard_fk`, and
`sobolev_fk`, allowing further plots without reloading the networks.

## Derivatives and normalizations

For each power model, the script differentiates the complete network prediction
(including input scaling, input PCA, output PCA and inverse output scaling)
with TensorFlow forward-mode automatic differentiation. Holding the other
inputs fixed also holds h fixed, so fixed k in h/Mpc corresponds to fixed
physical k during the redshift derivative.

If the network predicts R = P/P_norm, the physical growth is

```
f = -(1+z)/2 * [ (dR/dz)/R + (dP_norm/dz)/P_norm ].
```

Each model uses its own saved normalization table, interpolated with a cubic
spline (or lower degree if necessary). Target data are independently converted
to physical P and f using their FITS reference tables. The normalization for
stored FK is not assumed to equal growth derived from the power normalization.

Sobolev's public `eval_fk` interpolates its precomputed reference-growth table
linearly, whereas this independent reconstruction differentiates a spline of
the saved power reference. Therefore a small Sobolev consistency residual can
reflect reference interpolation, not a broken network derivative. This check
covers the emu_like saved model; it does not load a separately exported hi_fast
model or test its additional k interpolation.

On 128 randomly selected inputs by default, finite differences of **ln of
reconstructed physical P** check the automatic derivative at dz = 0.01, 0.003,
0.001. Central second-order differences are used inside the domain and
second-order one-sided differences at its boundaries. Configure these with
`--fd-samples` and `--fd-steps`. Results are absolute differences in growth,
not pass/fail assertions: ReLU boundaries and float32 cancellation can make the
smallest step less reliable than a larger one.

The diagnostic supports StandardScaler, LogStandardScaler, no scaling, and
saved sklearn PCA (including whitening). Other scaler types fail explicitly.
TF32 is disabled; CPU/GPU placement follows the environment. To force CPU,
prefix the command with `CUDA_VISIBLE_DEVICES=-1`. Memory use includes all
selected predictions; use a smaller `--max-samples` if needed.

`*_errors_vs_k.png` shows the 99th-percentile error at each k, with dotted
maximum-error curves. These help locate low-k tails that pooled histograms
can conceal.
