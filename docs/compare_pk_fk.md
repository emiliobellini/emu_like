# Comparing standard and Sobolev power/growth

## Compact per-model diagnostics

```bash
python scripts/check_train/plot_histograms.py \
  --roots /path/to/train --save-dir output/diagnostics
```

Each `pk_*` run (standard or Sobolev) produces five files:

- `summary_table.txt`, containing all observables and training information;
- `pk_accuracy.png` and `pk_errors_vs_k.png`;
- `fk_accuracy.png` and `fk_errors_vs_k.png`.

Standalone `fk_*` runs produce the summary and the two `fk` figures.
Standalone `cl_*` runs produce the summary, `cl_accuracy.png`, and
`cl_errors_vs_ell.png`, with multipole ℓ as the scale coordinate. All `cl_*`
spectra, including signed cross spectra, use relative errors. Zero truth
bins are undefined; errors close to a zero crossing can be large.
Files under `--save-dir` include the relative training-run path in their names.
The script no longer generates `histograms.png`, `worst_modes.png`, or the
separate growth histograms. Existing files from older runs are left intact;
use a fresh output directory for a clean collection.

Columns correspond to dataset ranges. Accuracy figures histogram one RMS
relative error per validation spectrum, expressed in percent, with shared
logarithmic axes and vertical lines at 0.01%, 0.05%, 0.1%, and 1%. Annotations
report evaluated/validation counts, the fraction above 1%, and exact zeros.
Exact zeros occupy the leftmost histogram bin but stay zero in statistics.
Scale figures have two rows: median errors with a 16th–84th percentile band
and a 95th-percentile curve above, and the three largest-RMS sample curves
below. Horizontal thresholds match the histogram thresholds. Scale errors
are absolute relative errors in percent; exact-zero curves are displayed
at 1e-8%. The same complete spectra enter both figures and the accuracy table.

Power runs derive growth through `fk_from_pk`. Sobolev growth figures overlay
`eval_fk` with a consistent contrasting color and dashed line style.
There are no additional consistency figures or CLASS recomputations.
The summary contains median, 95th percentile, and maximum RMS relative
errors, threshold exceedance fractions, and validation/evaluated/excluded
counts for every range and method. It also contains physical absolute-growth
RMS/maximum errors, Sobolev consistency RMS/maximum absolute differences,
training history, and elapsed diagnostic time per range and observable.
Absolute-growth and consistency statistics include all finite bins, retaining
information near zero growth even when relative errors are undefined.

The validation split is reconstructed from each run's `params.yaml`: apply
its training finite-row filter (including growth for Sobolev), join files
in the saved order, and split once using `frac_train` and
`train_test_random_seed`. Original FITS files must be unchanged since training;
an unseeded split cannot be reconstructed. Predictions and targets are
restored to physical units using their respective saved reference tables.
A spectrum is excluded from relative diagnostics if any bin is undefined.
For growth, `--growth-floor` (default `1e-6`) excludes bins with physical
`|f_data|` at or below the floor. Power and angular spectra exclude zero truth.

Use `--growth-batch-size` (default 128) to control derivative memory.
The existing `--skip-growth-histograms` option skips both derived-growth
figures for power runs; growth summary tables are still computed.
The matching FITS growth and reference extensions are needed for power runs.

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
