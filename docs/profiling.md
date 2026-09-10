# Profiling Sobolev training on the cluster

Notes from 10 September 2026, using Python 3.12, TensorFlow 2.19.0 and an
NVIDIA A100-PCIE-40GB. Run repository commands from
`/ceph/hpc/home/bellinie/emu_like`.

## Capture now, inspect later

The training log reports overall throughput (`ms/step` or `s/step`). A profile
records CPU operations, GPU kernels, transfers and compilation to explain
where time goes.

Capture happens **during training**, on the allocated compute node. You can
inspect the saved trace **after the job finishes**, using TensorBoard on a
login node. Viewing a saved GPU trace does not require a GPU allocation.

A trace already written to disk survives a later timeout: our original job
timed out, but its capture of batches 20–25 remained available.

## Install the packages

### Use the existing environment

The shared venv has already been repaired and verified. These commands are
for reproducing its relevant package versions or reinstalling the code:

```bash
cd /ceph/hpc/home/bellinie/emu_like
source /ceph/hpc/home/bellinie/venv/bin/activate
python -m pip install -r docs/profiling-requirements.txt .
python -m pip check
```

The expected check result is `No broken requirements found.` The adjacent
[requirements file](profiling-requirements.txt) includes TensorBoard, its
profiling plugin, XProf and compatible dependency versions. It is not a full
lock file for the entire environment.

For later installations of repository changes, retain these constraints:

```bash
python -m pip install -c docs/profiling-requirements.txt .
python -m pip check
```

`pip install .` installs a snapshot of the package. Reinstall after changing
source code, or use the source directory through `PYTHONPATH`, as the batch
script below does.

### Optional separate environment

If you prefer a separate environment, create one before the installation
commands above:

```bash
module load Python/3.12.3-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0
python -m venv /ceph/hpc/home/bellinie/venv-profile
source /ceph/hpc/home/bellinie/venv-profile/bin/activate
```

Use this activation path in TensorBoard commands and, if training there too,
in the batch script. The original repaired venv was tested; a fresh install
of this optional environment has not been separately tested.

### The protobuf conflict we encountered

TensorFlow 2.19 requires **protobuf <6**. Installing profiling dependencies had
left protobuf 7.36.1 in the environment, causing `MessageFactory ... GetPrototype`
messages. Reinstalling `emu_like` selected protobuf 5.29.6, satisfying
TensorFlow but conflicting with six newer Google Cloud/gRPC packages requiring
protobuf >=6.33.5.

We repaired this by selecting compatible older versions of those six packages,
recorded in the requirements file. TensorFlow stayed at 2.19.0. Upgrading
protobuf alone would recreate the TensorFlow conflict.

An installation can succeed while leaving other installed packages
incompatible. Follow it with
[`python -m pip check`](https://pip.pypa.io/en/stable/cli/pip_check/).
Restart TensorBoard and Python processes after package changes.

After the repair, dependency checks, small CPU training steps with fk off/on,
and conversion of the saved GPU trace to Overview and Trace Viewer data passed.

## Configure profiling

Copy a **complete** existing Sobolev YAML file, preserving its architecture
and dataset settings. Change these fields in the copy:

```yaml
output:
  path: output/my_sobolev_profile/
emulator:
  name: sobolev_ffnn_emu
  args:
    epochs: 1
    batch_size: 512
    profile_batches: [20, 25]
    # Optional; defaults to output.path/tf_profile:
    # profile_log_dir: output/my_sobolev_profile/tf_profile
```

This is a configuration fragment, not a complete training configuration.
Choose a new output directory for each experiment. Do not add force or resume
flags to a fresh profiling test.

The repository supports these options for `sobolev_ffnn_emu`. The inclusive
range activates the Keras TensorBoard callback. Starting at batch 20 avoids
much of the initial setup, although profiler startup overhead may still be
visible. Omit `profile_batches`, or set it to `null`, to disable capture.
Profiling options may change on strict or warm resume.
See the [Keras callback reference](https://keras.io/api/callbacks/tensorboard/).

### Pk-only versus fk training

Our normal configuration has two Pk-only warm-up epochs and a 2000-epoch fk
ramp. Early batches of a fresh run therefore measure **Pk-only training**.

To exercise the derivative branch immediately, set these values under
`emulator.args` in a test copy:

```yaml
pk_weight: 1.0
fk_weight: 1.0
fk_warmup_epochs: 0
fk_ramp_epochs: 0
```

This enables full fk weight from the first batch. It is a performance test,
not a change to the production schedule or evidence of model convergence.

## Submit a GPU test

Save the following as `run_profile_gpu.sh` in the repository. Replace the
final YAML path with your complete test configuration. The module versions
and resources below match the successful tests on the date of these notes.

```bash
#!/bin/bash
#SBATCH --job-name=profile_sobolev
#SBATCH --partition=dev
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/o%j.%x
#SBATCH --error=logs/e%j.%x

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load Python/3.12.3-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.10.2.21-CUDA-12.6.0
source /ceph/hpc/home/bellinie/venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); print(g); assert g, 'No GPU visible'"
python main.py train init_files/train/my_profile.yaml -v
```

Submit from the repository root:

```bash
mkdir -p logs
sbatch run_profile_gpu.sh
```

Replace `JOBID` with the number returned by Slurm:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS -P
```

`COMPLETED` and exit code `0:0` indicate a successful exit. Check both stdout
and stderr. The message `Collecting XSpace to repository` gives the saved
capture path. GPU profiling uses CUPTI, available through the CUDA environment
in our runs; see [TensorFlow's profiler prerequisites](https://www.tensorflow.org/guide/profiler#install_the_profiler_and_gpu_prerequisites).

Our local comparison scripts are in `init_files/train/lcdm_k_sobolev/`:
`run_pk_m_no_xla_comparison_gpu.sh` and
`run_pk_m_fk_no_xla_comparison_gpu.sh`. These experiment files and outputs
may be ignored by Git; the guide does not depend on their presence.

## View the saved profile

On a login node:

```bash
cd /ceph/hpc/home/bellinie/emu_like
source /ceph/hpc/home/bellinie/venv/bin/activate
tensorboard --logdir output/sobolev_fk_no_xla_comparison/tf_profile \
  --host localhost --port 6006
```

Change `--logdir` to your experiment. Leave TensorBoard running. Ctrl+C stops
the viewer server; the saved profile remains on disk.

With **VS Code Remote SSH**, open the **Ports** panel, forward **6006**, and
open the forwarded address in your browser.

Alternatively, run this from your **local computer**:

```bash
ssh -N -L 6006:localhost:6006 bellinie@CLUSTER_LOGIN_HOST
```

Replace `CLUSTER_LOGIN_HOST` with an SSH host that reaches the **same login node**
running TensorBoard. A cluster alias that assigns a different node will not
reach that server; use your normal SSH routing to the specific node.
Open `http://localhost:6006` and keep both the tunnel and TensorBoard running.

Select **Profile**, the run (usually `train`), and the capture timestamp.
For example, the fk test saved:

```text
tf_profile/train/plugins/profile/2026_09_10_11_43_11/
    vglogin0007.vega.izum.si.xplane.pb
```

## Read the performance data

Start with **Overview Page**, then investigate slow intervals in **Trace
Viewer**. **TensorFlow Stats** and **GPU Kernel Stats** identify expensive
operations. See the [profiler guide](https://www.tensorflow.org/guide/profiler).

Expand `/host:CPU` and `/device:GPU:0` and zoom into one batch. At a 20-second
scale, tiny GPU events can appear as isolated vertical lines. Collapse
unrelated sections or take multiple screenshots when the timeline is too tall.
The saved `.xplane.pb` also supports direct analysis without screenshots.

Lessons from our profiles:

- **Steps** bars mark intervals, not continuous GPU activity.
- **All Others** is unclassified time. Our old overview missed compilation
  clearly visible in the raw trace; it was not a reliable CPU/GPU breakdown.
- Operation placement percentages are not percentages of elapsed time.
- The original overview averaged mixed events and reported 1.2 seconds;
  actual `Profiled batch` events averaged 3.37 seconds.
- CPU events can be nested. Adding parent and child durations double-counts time.
- Summed GPU kernel durations are not hardware utilization percentages.
- Compare steady batches separately from initialization, profiling overhead,
  validation and saving. The final Keras epoch summary and the selected six
  batches measure different intervals.

An optional live utilization check, where overlapping Slurm steps are allowed:

```bash
srun --jobid=JOBID --overlap --ntasks=1 --cpus-per-task=1 \
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
  --format=csv -lms 500
```

Stop the monitor with Ctrl+C. This requires the job to be running and cannot
reconstruct historical utilization afterward.

## XLA and the performance fix

XLA is an optional compiler that optimizes groups of TensorFlow operations.
Compilation costs time, and the resulting code is normally reused. See the
[OpenXLA architecture](https://openxla.org/xla/architecture).

Our original GPU trace showed backend compilation repeating every batch.
All three Sobolev compile paths—fresh, strict resume and warm resume—now use:

```python
self.model.compile(optimizer=optimizer, jit_compile=False)
```

This disables XLA for the training function while preserving TensorFlow graph
execution and GPU use. It does not set `run_eagerly=True` or force CPU execution.

The exact recompilation trigger remains unknown. Batch size was fixed, and
the outer function's trace count stayed at one. The conditional derivative
branch (`tf.cond`, `batch_jacobian`, second-order gradients) is a suspect,
not a proven cause. Compiler diagnostics would be needed to explain the
failure to reuse compiled code.

In our installed Keras 3.10, automatic JIT selection already disables XLA on
CPU-only machines. Explicitly disabling it probably does not change those CPU
runs. XLA can help other workloads; disabling it is not a universal optimization.

## Results from our tests

All captures used batches 20–25, batch size 512 and an A100 40 GB. The network
had two hidden layers of 1024 neurons and 600 outputs.

| Job | Phase | XLA | Mean profiled batch | Outcome |
| --- | --- | --- | ---: | --- |
| 44758152 | Pk-only warm-up | On | 3367.59 ms | Hit 30-minute limit before finishing the epoch; trace saved |
| 44761517 | Pk-only warm-up | Off | 4.32 ms | One epoch completed; 66 seconds total job time |
| 44762090 | Pk + fk, both weights 1 | Off | 62.24 ms | One epoch completed; 97 seconds total job time |

The captured Pk-only batches became about **780× faster**. With XLA off,
Pk+fk batches were about **14× slower** than Pk-only. Training plus validation
took about 11 seconds for the Pk-only test and 44 seconds for the fk test;
total job time also includes setup and data loading.

The old trace showed about 1.05 seconds of backend compilation per batch,
plus substantial CPU transpose and segment-sum work. The two new captures
had no repeated XLA backend compilation events.

These are short performance tests, not convergence comparisons. There is no
XLA-on fk measurement, so the 780× factor must not be applied to fk or CPU runs.

Local extracted measurements:

- `output/sobolev_pk_m/profile_analysis_44758152.json`
- `output/sobolev_pk_m_no_xla_comparison/profile_comparison.json`
- `output/sobolev_fk_no_xla_comparison/profile_analysis.json`

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| CUDA warnings when viewing on a login node | Saved profiles can be viewed without a GPU. Check for the final serving URL. |
| No GPU inside the training job | Check allocation, modules and the TensorFlow GPU assertion. |
| `MessageFactory ... GetPrototype` | Check protobuf compatibility; use the pins and run `pip check`. |
| Successful installation with dependency conflicts | The environment still needs repair; do not blindly upgrade protobuf. |
| Profile dashboard missing | Verify the plugin in the viewer's venv, restart TensorBoard and check the capture/log directory. |
| Trace looks empty | Expand CPU/GPU sections and zoom into the thin events. Check whether GPU events were captured. |
| Browser cannot connect | Keep TensorBoard running and forward to the same login node. |
| Job finishes after one epoch | Expected for these tests; check exit status and final save messages. |

Production runs can retain their intended fk warm-up and ramp with the XLA
fix in place. Disable profiling once the performance measurements are complete.
