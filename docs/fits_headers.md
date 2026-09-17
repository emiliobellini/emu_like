# Sampling FITS headers

`PRIMARY` keeps the input sampling configuration (`params`, `x_sampler`, and
`y_model`). It is not a literal YAML document: comments and execution options
are not stored. Runtime additions do not belong here. Emulator artifacts have
their own primary-header schema.

`X_DATA.parameters` is an ordered dictionary keyed by parameter name. Each
entry has a zero-based `column` and the original `prior`, including configured
`min`/`max` bounds when supplied. These are prior ranges, not observed extrema.

`PK_*` and `FK_*` retain their spectrum/grid metadata and additionally store:

- `class_parameters`: fixed effective CLASS arguments, including automatic
  output additions, excluding sampled parameters.
- `class_metadata`: schema version, provenance, and the rule overriding
  `z_max_pk` for each row. Combine the fixed arguments with the columns in
  `X_DATA`, then apply this rule to reconstruct the explicit CLASS inputs.
  Growth outputs require derivative coverage even when examining a different
  spectrum from the same calculation.
- `stored_quantity` and, for ratios, `stored_units = dimensionless`. Existing
  spectrum units and formula fields describe the underlying physical spectrum.

`REF_*` contains only the dictionary passed to CLASS for the reference
cosmology. Loading restores this dictionary and the saved grids. For outputs
with `ratio=False`, reference arrays contain ones, not CLASS predictions.
Reproducing references requires the same CLASS implementation/defaults and the
spectrum extraction and grid conventions; the parameter dictionary alone does
not identify the historical CLASS binary. Reference headers are checked for
agreement when loading.

Grid HDUs and completion masks have no custom headers. Their arrays contain the
grid values or mask.

## Updating existing sampling files

From the checkout, preview changes without running CLASS:

```bash
PYTHONPATH=src python scripts/check_data/update_fits_headers.py /path/to/sample.fits
```

Apply to one or several files:

```bash
PYTHONPATH=src python scripts/check_data/update_fits_headers.py --apply /path/to/sample.fits
```

The script keeps a full `sample.fits.before_header_update.bak` first, refuses to
overwrite that backup, and preserves primary/reference headers and all arrays.
It uses the sampling writer lock; run it after sampling jobs finish. Files
already carrying the new metadata are skipped. Ensure space for the backup
and FITS rewriting when headers grow.

Legacy runtime arguments are inferred from the input and current code and
labelled `inferred_from_input_and_current_code`. This cannot establish which
unrecorded runtime modifications an older generator actually used. Reference
headers are preserved rather than guessed. New sampling uses
`recorded_at_generation`; joined models use `reconstructed_for_join` because
their combined calculation limits need not equal each original row's limits.
Loaded metadata retains its original provenance when saved again.
