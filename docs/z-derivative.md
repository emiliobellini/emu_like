# Z-only Sobolev derivative: implementation and validation

## Calculation

`SobolevTrainingModel._pk_and_z_derivative` uses
[`tf.autodiff.ForwardAccumulator`](https://www.tensorflow.org/api_docs/python/tf/autodiff/ForwardAccumulator)
with a tangent that is one in the scaled-redshift coordinate and zero elsewhere.
The Jacobian-vector product gives every output's derivative with respect to
scaled z, without creating the full per-sample Jacobian.

For the benchmark network, this replaces an explicit `[512, 600, 6]` Jacobian
with a `[512, 600]` directional derivative and avoids large intermediate
reverse-mode tensors. This is automatic differentiation, not a finite-difference
approximation.

The derivative stays inside the outer weight tape. Consequently the optimizer
still receives the mixed derivatives required by the fk loss. The physical
conversion is unchanged:

```text
d(log Pk ratio)/dz = pk_scale / z_scale * d(pk_scaled)/d(z_scaled)
fk = reference_growth(k,z) - (1+z)/2 * d(log Pk ratio)/dz
```

Both training/validation losses and `eval_fk` use the new helper. Batch
normalization and dropout remain forbidden by the Sobolev architecture:
the calculation assumes each sample is independent of the other batch rows.
The Pk-only warm-up path and checkpoint format are unchanged. XLA remains off.

## Automated regression tests

From the repository root, with the training venv active:

```bash
PYTHONPATH=src python -m unittest discover -s tests \
  -p test_sobolev_derivatives.py -v
```

The full-Jacobian implementation is retained as a test-only reference subclass.
Six tests cover:

- All four supported smooth activations; z in different input columns.
- Pk, z derivatives, physical fk, losses and all weight gradients, including
  fk-only gradients.
- Graph execution with variable batch size and sample independence.
- Float64 central differences at three step sizes for the z derivative and
  a weight-direction derivative of a derivative-dependent loss.
- Actual Adam updates across fk weights 0, 0.001 and 1, including optimizer state.
- Public `eval_fk` for batched, single and dictionary inputs, plus an analytic
  linear-network check of redshift scaling and physical fk conversion.

All six passed on CPU in the installed TensorFlow 2.19 environment.

## Full-size accuracy checks

Separate cluster jobs compared the two methods on the actual 600-output
network using both fresh weights and the saved checkpoint from job 44762090.
Each held-out batch combined 128 low-z points, 128 high-z points and 256 other
held-out points. Tests covered training and inference modes, and gradients
with Pk weight zero and one. Ten paired updates used identical training batches,
identical initial weights and fresh identical Adam states.

Direct comparisons initially used `atol=1e-6, rtol=1e-4`. For standardized fk,
the absolute tolerance is propagated from physical fk as
`1e-6 / growth_scale`, rather than applying the same absolute number in
different units.

On the full-float32 GPU check:

- All direct derivative, loss and weight-gradient comparisons passed.
- The largest absolute weight-gradient difference was about `5.6e-8`.
- One weight tensor after the first Adam step exceeded the initial elementwise
  tolerance near zero: maximum absolute difference `2.43e-6`.
- After ten updates, held-out Pk predictions differed by at most `6.92e-6`,
  with relative L2 error about `5.3e-7`.
- Held-out scaled Pk MSE: `0.19760299` (JVP), `0.19760288` (reference).
- Held-out scaled fk MSE: `0.16948833` (JVP), `0.16948827` (reference).

The CPU check passed direct derivatives, losses, gradients and the first update.
After ten updates, one held-out Pk comparison exceeded the initial absolute
tolerance near zero (`6.92e-6` maximum, `3.1e-7` relative L2).
Held-out Pk MSE agreed at printed precision (`0.19760293`); fk MSE was
`0.16948783` versus `0.16948788`.

The original strict reports are preserved, including their failures. The
separate numerical assessment accepts these optimizer/short-run exceptions
only when **both** maximum absolute error and relative L2 error are <=`1e-5`.
This accounts for accumulated float32 rounding and optimizer sensitivity;
the direct derivative and gradient checks retain their original tolerances.
It is not a claim of bitwise equality or of identical long-term convergence.

## GPU precision matters

The first full-size GPU check with default TF32 enabled failed the strict
derivative comparison: maximum absolute difference about `9.4e-5`, relative
L2 difference about `4.2e-4` (0.042%). Disabling TF32 reduced the derivative
disagreement to ordinary float32 rounding levels.

TF32 permits reduced-precision inputs to some float32 matrix multiplications.
The Jacobian and JVP use different matrix shapes and may therefore accumulate
different rounding errors. See
[TensorFlow's TF32 documentation](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_tensor_float_32_execution).

For strict comparisons, set this **before tracing or training** in the same
Python process:

```python
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
```

The repository's `main.py` now disables TF32 at startup, before dispatching
to any pipeline, and prints `TF32 enabled: False`. Runs launched through this
entry point use the validated precision setting. Direct library users and
notebooks must set it explicitly in their own Python process using the code
above; the model class does not change this process-wide setting itself.
The strict full-model accuracy assessment applies to TF32-disabled execution;
default-TF32 equivalence at those tolerances has not been established.

## Controlled GPU benchmark

Job 44764416 ran each implementation in a separate process on an A100 40 GB,
with TF32 and XLA disabled for both. Batch size was 512, both loss weights
were one, and the mean covered profiled batches 20–25 after warm-up.

| Measurement | Full Jacobian | Z-only JVP |
| --- | ---: | ---: |
| Mean training batch | 200.74 ms | 6.06 ms |
| Peak TensorFlow GPU allocation during capture | 10.28 GiB | 1.41 GiB |

This is about **33 times faster**, with about **7.3 times lower peak GPU
allocation**. The memory figures include resident model/data allocations and
were measured using `get_memory_info`, after resetting peak statistics before
batch 20; they are not total process memory from `nvidia-smi`.

The previous 62.24 ms GPU baseline allowed TF32, so it is not the matched
precision baseline for this table. First-call tracing, data loading, validation
and saving are excluded from the per-batch figures.

### CPU benchmark

Job 44764545 measured the z-only implementation on node `cn0430`, using 16
intra-operation threads and one inter-operation thread, batch size 512 and
XLA disabled. Its six profiled batches averaged **52.93 ms**. The earlier
full-Jacobian CPU run (44763332, same node and thread settings) averaged
**7733.93 ms**, approximately **146 times slower**. These are separate short
runs, not a statistical estimate across many jobs; weights and batch ordering
were not matched between the earlier CPU baseline and the new benchmark.

## Local experiment artifacts

The following files live under `output/z_derivative_validation/` and may be
ignored by Git:

- `unit_tests.log`: regression test output.
- `gpu_accuracy.json`: the initial TF32-enabled failure.
- `cpu_checked_accuracy.json`, `gpu_checked_accuracy.json`: complete strict
  accuracy reports, including the small accumulated-update discrepancies.
- `accuracy_assessment.json`: explicit assessment of those discrepancies.
- `gpu_reference_timed/timing.json`, `gpu_jvp_timed/timing.json`: controlled
  GPU timing and memory measurements.
- `cpu_jvp_timed/timing.json`: CPU directional-derivative timing.
- `benchmark.py`, `validate_checked.py`, and the Slurm scripts/configurations:
  experiment harnesses. These import the numerical oracle from the tests.

The benchmark jobs explicitly set `PYTHONPATH` to the source checkout. An
existing non-editable package installation still needs reinstalling to use
these source changes in ordinary training commands.

## Reference-growth correction (2026-09-11)

The Sobolev network predicts the log of `P / REF_PK`. Its physical growth
prediction therefore uses

```
f = -0.5 * (1 + z) * d(REF_PK)/dz / REF_PK
    -0.5 * (1 + z) * d(log(P / REF_PK))/dz
```

`REF_FK` is only the normalization used to store the fk target; it is not
necessarily physical reference growth. It can contain ones. The dataset loader
now differentiates a cubic spline of `REF_PK` on `Z_ARRAY` to construct the
reference term. A constant Pk normalization correctly contributes zero.
Stored `FK` targets are multiplied by their interpolated `REF_FK` normalization
before fitting the growth scaler, so the loss always compares physical growth.

Previously saved Sobolev checkpoints may contain an incorrect reference and
were optimized against that incorrect objective. They are rejected on load;
use a fresh output directory for corrected training. Existing exports and
already-running jobs are not repaired by this source change.

This correction does not remove or repair the extreme finite fk spikes found
in the lcdm_k sample files. Their generation still requires investigation.
