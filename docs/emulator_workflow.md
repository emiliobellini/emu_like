# Emulator generation workflow

This guide describes the complete workflow for building a cosmological
emulator: sample and train on Vega, archive the results on Meteo, export the
emulator, and install the exported files in `hi_fast`.

Run Vega commands from the root of the `emu_like` repository unless stated
otherwise. Replace `lcdm` in the examples with the model being built, such as
`lcdm_k`, `lcdm_nu`, or `lcdm_nu_k`.

## Paths used in this guide

Define the model and the machine-specific roots once at the start of a Vega
shell session:

```bash
export MODEL="lcdm"
export EMU_REPO="$(git rev-parse --show-toplevel)"
export VEGA_DATA_ROOT="/ceph/hpc/data/s25r06-05-users"
export MODEL_DATA="${VEGA_DATA_ROOT}/${MODEL}"
export METEO_MODEL_ROOT="/d4/CAC/ebellini/data/emu_like/${MODEL}"

cd "${EMU_REPO}"
mkdir -p logs
```

The shell variables make the commands below reusable. They do **not** change
the paths written by the two configuration-generator scripts; review their
settings as described in the next sections.

The workflow moves data through these locations:

```text
Vega working data                ${VEGA_DATA_ROOT}/${MODEL}/
        | emu_sync push
        v
Meteo archive                    /d4/CAC/ebellini/data/emu_like/${MODEL}/
        | hi_fast_sync (run locally)
        v
hi_fast exported emulators      <hi_fast checkout>/emu/${MODEL}/
```

## 1. Generate the samples on Vega

### Create the sampling configuration

Before running the generator, edit the `Settings` block in
[`scripts/get_ini/get_ini_files_sample.py`](../scripts/get_ini/get_ini_files_sample.py).
In particular, check:

- `model`;
- `n_samples_1000` (the actual number of samples is 1,000 times this value);
- `timeout` and `save_interval`;
- `data_root`;
- the sampled parameter ranges, CLASS arguments, and the `k` and `ell` grids.

The script currently also embeds the Vega repository, virtual-environment,
and email paths in the generated Slurm files. Update those template values if
the repository or environment has moved.

Generate the YAML and Slurm files:

```bash
python scripts/get_ini/get_ini_files_sample.py
```

Review the generated files under `init_files/sample/${MODEL}/` before
submitting them. Running the generator again overwrites files with matching
names.

### Submit or resume sampling

Submit each required parameter-space and spectrum combination. For example:

```bash
sbatch "init_files/sample/${MODEL}/run_cl_100_ext.sh"
```

The generated sampling jobs call `main.py sample` with `--force`. If the
output does not exist, this starts a new sample; if it does exist, sampling is
resumed. It is therefore normal to submit the job again after a timeout until
the requested sample is complete.

Monitor jobs and logs with:

```bash
squeue --me
tail -f logs/o<JOB_ID>.<JOB_NAME>
tail -f logs/e<JOB_ID>.<JOB_NAME>
```

### Run independent row ranges

Regenerate the Slurm files with the updated sampling generator to enable
optional `START_ROW STOP_ROW` arguments. With no arguments, the generated
scripts keep the normal sampling/resume behaviour described above.

Range jobs read the same main FITS and compute their assigned rows serially,
writing separate numbered FITS files in its folder. For a **new** dataset,
prepare the input table and reference spectra once, in your sampling Python
environment, before submitting any range jobs:

```bash
PYTHONPATH=src python main.py sample \
    "init_files/sample/${MODEL}/cl_100_ext.yaml" --prepare-only
```

Skip preparation if the main FITS already exists. Range jobs use the settings,
input rows and reference data stored in that FITS; the YAML supplies the output
path, timeout and checkpoint interval.

Submit separate jobs with explicit ranges:

```bash
sbatch "init_files/sample/${MODEL}/run_cl_100_ext.sh" 0 1000
sbatch "init_files/sample/${MODEL}/run_cl_100_ext.sh" 1000 2000
```

These evaluate rows **0–999** and **1000–1999**: indices are zero-based and the
stop is exclusive. Both arguments are required when selecting a range. The
generated scripts pass them as `--start-row` and `--stop-row`, and import the
code from this checkout using `PYTHONPATH`.

Once execution starts, the first job is renamed to
`sample_lcdm_cl_100_ext_rows_0_1000` (for `MODEL=lcdm`). While pending it retains
the base name, because Slurm does not expand script arguments in `#SBATCH`
directives. Log filenames also retain the original base name, for example
`logs/o<JOB_ID>.sample_lcdm_cl_100_ext`; the job ID distinguishes the files.

Each numbered FITS is also its checkpoint, containing original row IDs and
all corresponding outputs. For these examples the files are:

```text
${MODEL_DATA}/sample/cl_100_ext.rows-000000000-000001000.fits
${MODEL_DATA}/sample/cl_100_ext.rows-000001000-000002000.fits
```

Rerun the same `sbatch` command after a timeout or node failure to resume the
range from its saved checkpoint. A killed job may lose work since its last
save, but later results do not shift to different input rows. Checkpoints use
the YAML's `output.save_interval` (100 rows if unspecified). Do not run two
jobs with the identical range at once; their shared checkpoint is locked.

### Monitor saved sampling progress

The status script reads the main FITS and its numbered checkpoints directly.
It works for ordinary sampling and independent range jobs, without requiring
progress bars in the Slurm logs. Show one compact summary row per dataset:

```bash
PYTHONPATH=src python scripts/check_data/check_sampling_status.py \
    "${MODEL_DATA}/sample"
```

Alternatively, pass a particular main FITS or sampling YAML. Add `--ranges`
to show the saved count and percentage for each numbered checkpoint:

```bash
PYTHONPATH=src python scripts/check_data/check_sampling_status.py \
    "init_files/sample/${MODEL}/cl_100_ext.yaml" --ranges
```

The summary columns mean:

| Column | Meaning |
| --- | --- |
| `TOTAL` | Number of input rows in the main FITS. |
| `MAIN` | Completed rows already stored in the main FITS. |
| `UNMERGED` | Additional unique rows saved in checkpoints but absent from the main FITS. |
| `SAVED` | Unique completed rows across the main FITS and checkpoints. |
| `%` | `SAVED` as a percentage of `TOTAL`. |
| `MISSING` | Input rows without a saved result. |
| `FILES` | Number of readable, structurally valid range checkpoints. |

Overlaps count only once in the summary. `UNMERGED` excludes checkpoint rows
that would overwrite an already completed main row, so it can be zero even
when files still need merging. The per-range table shows each file's own
progress, including overlapping rows. Counts include computed NaN results.

**This reports saved work, not unsaved computations in a running process.**
Counts advance when a checkpoint is saved. A range has no visible entry before
its first checkpoint, and an incomplete checkpoint alone does not tell you
whether its job is running or stopped; use `squeue --me` for job state.
For periodic refreshes from the repository root:

```bash
PYTHONPATH=src watch -n 60 python scripts/check_data/check_sampling_status.py \
    "${MODEL_DATA}/sample"
```

Sampling stdout logs remain accepted as inputs when they contain a FITS path.
New range jobs print that path immediately on startup; for older range jobs
whose logs do not contain it yet, pass the FITS, YAML, or directory instead.
The report reads only headers, completion masks and row IDs. Dataset identity
and full output checksums are validated during merging, not by this progress
report. Warnings identify unreadable or inconsistent files; retry if a legacy
writer was saving or a merge was removing checkpoints during the scan.

### Merge range results into the main FITS

Wait until the selected range workers and any ordinary writer to the main
FITS have stopped, then merge the range files explicitly:

```bash
PYTHONPATH=src python main.py sample \
    "init_files/sample/${MODEL}/cl_100_ext.yaml" \
    --merge-ranges \
    "${MODEL_DATA}/sample/cl_100_ext.rows-000000000-000001000.fits" \
    "${MODEL_DATA}/sample/cl_100_ext.rows-000001000-000002000.fits"
```

The merge validates the dataset identity and row IDs, then copies results into
their original positions in the main FITS. Files are applied in the listed
order: **later files overwrite overlapping rows**, including NaN results.
Existing rows outside the merged ranges are preserved. A stopped, partially
completed range can also be merged; only its saved rows are copied.

The main FITS is saved through a verified temporary file and atomic replacement.
Only after successful saving are the supplied range files/checkpoints deleted.
Add `--keep-ranges` to retain them. Allow disk space for a second main FITS and
memory for the full dataset during merging. Small `.lock` files remain
intentionally; they are not checkpoints and do not indicate an active job.

The `SAMPLE_DONE` extension records which rows have been computed. Missing rows
contain NaNs, while a computed NaN still counts as completed. After merging,
ordinary `--resume` (or a generated job with no arguments) fills missing rows
serially using this mask. Before the first merge, legacy FITS files still
resume by appending after the saved prefix. Complete and merge the required
ranges before validating the full sample.

**Already running legacy jobs can continue while range jobs compute, but must
finish or stop before merging into their main FITS.** They do not honor the
new locks. If a range job fails while initially reading a legacy writer's
in-place save, retry after that save finishes. After the first merge, all
writers must use the updated code so they understand the completion mask.

### Validate the samples

Submit the appropriate check job, for example:

```bash
sbatch "init_files/sample/checks/${MODEL}_cl.sh"
```

Alternatively, validate an individual FITS file interactively:

```bash
python tests/sampler.py \
    "${MODEL_DATA}/sample/cl_100_thin.fits" \
    --n_rands 1000
```

The check recomputes randomly selected spectra and compares them with the
stored values. Use `--threshold VALUE` to change the comparison tolerance.

After validation, preserve the relevant Slurm output and error logs with the
model data. Select the files explicitly so logs from other runs are not moved
accidentally:

```bash
mkdir -p "${MODEL_DATA}/logs/sample"
mv logs/o<JOB_ID>.<JOB_NAME> logs/e<JOB_ID>.<JOB_NAME> \
    "${MODEL_DATA}/logs/sample/"
```

## 2. Train the emulators on Vega

### Create the training configuration

Edit the `Settings` block in
[`scripts/get_ini/get_ini_files_train.py`](../scripts/get_ini/get_ini_files_train.py).
Check at least:

- `model`, `n_samples_1000`, and `data_root`;
- network size, learning rate, batch size, and patience;
- timeout, PCA settings, scalers, and loss settings;
- the repository and virtual-environment paths in the Slurm template.

Then generate and review the configurations:

```bash
python scripts/get_ini/get_ini_files_train.py
```

The generated files are stored under `init_files/train/${MODEL}/` and files
with matching names are overwritten.

### Submit or resume training

Submit one job for each required output, for example:

```bash
sbatch "init_files/train/${MODEL}/run_cl_TT_lensed.sh"
```

The generated jobs use strict resume mode (`--resume-strict --force`). An
empty output directory starts a new training; a non-empty one resumes the
stored optimization problem. Submit the job again after a timeout until
training is complete. Use `--resume-warm` manually only when intentionally
loading the stored weights while taking the datasets, preprocessing, loss,
and training policy from a new parameter file.

### Inspect training

Plot loss histories:

```bash
python scripts/check_train/plot_loss.py --roots "${MODEL_DATA}/train"
```

Plot error histograms:

```bash
python scripts/check_train/plot_histograms.py --roots "${MODEL_DATA}/train"
```

While jobs are still running, it is often more convenient to collect plots in
one separate directory:

```bash
mkdir -p "${MODEL_DATA}/diagnostics"
python scripts/check_train/plot_loss.py \
    --roots "${MODEL_DATA}/train" \
    --save-dir "${MODEL_DATA}/diagnostics"
python scripts/check_train/plot_histograms.py \
    --roots "${MODEL_DATA}/train" \
    --save-dir "${MODEL_DATA}/diagnostics"
```

As with sampling, move the selected training logs into a dedicated folder
after inspecting them:

```bash
mkdir -p "${MODEL_DATA}/logs/train"
mv logs/o<JOB_ID>.<JOB_NAME> logs/e<JOB_ID>.<JOB_NAME> \
    "${MODEL_DATA}/logs/train/"
```

## 3. Archive the model on Meteo

Meteo is reached through the Ajdovscina jump host:

```bash
ssh -J ebellini@ajdovscina.ung.si ebellini@meteo.ung.si
```

Create the archive root once if it does not exist:

```bash
ssh -J ebellini@ajdovscina.ung.si ebellini@meteo.ung.si \
    "mkdir -p '${METEO_MODEL_ROOT}'"
```

From Vega, synchronize the model after sampling, training, and export:

```bash
emu_sync push "${MODEL}"
```

To restore or update Vega from the Meteo copy:

```bash
emu_sync pull "${MODEL}"
```

To synchronize just one subdirectory, add its path relative to the model:

```bash
emu_sync check "${MODEL}" emulator_files
emu_sync push "${MODEL}" emulator_files
emu_sync pull "${MODEL}" emulator_files
# Nested directories are supported too:
emu_sync push "${MODEL}" logs/train
```

The selected directory keeps the same location inside the model on both
machines; other model subdirectories are not transferred. Omitting `SUBDIR`
synchronizes the whole model. Absolute paths and `..` components are rejected.
Missing destination subdirectories are created (the archive root must exist).

For partial recomputation on Vega, push the updated files first, then pull
the same selection to restore files held only on Meteo. Transfers overwrite
differing destination files even if the destination is newer. The existing
`check` mode uses `--update`, so it skips newer destination files and is not
an exact preview of those overwrites.

`rsync` is incremental, so repeating these commands transfers only changed
content. The trailing slashes in the helper definitions mean “copy the
contents of this model directory,” avoiding an extra nested model directory.
These commands do not use `--delete`; files found only at the destination are
left in place.

## 4. Export the emulator on Vega

Export all trained outputs into one deployment directory:

```bash
python main.py export \
    --input "${MODEL_DATA}/train" \
    --output "${MODEL_DATA}/emulator_files" \
    --verbose
```

The export destination must be empty. If it already contains an export, move
or remove that old directory deliberately before rerunning the command. Check
the exported `.joblib` and `.keras` files, then archive the updated model:

```bash
emu_sync push "${MODEL}"
```

## 5. Install the export in `hi_fast`

Run this step on the machine containing the `hi_fast` checkout (normally the
laptop), not on Vega. After installing the portable helper from the next
section, set the local root and pull the selected model:

```bash
export HI_FAST_EMU_ROOT="${HOME}/Codes/hi_fast/emu"
export MODEL="lcdm"
hi_fast_sync "${MODEL}"
```

This copies the **contents** of Meteo's `emulator_files/` directory into
`${HI_FAST_EMU_ROOT}/lcdm/`.

## Portable synchronization helpers

The current functions can be made reusable across Vega and the laptop by
putting the following definitions in `~/.bashrc` on the relevant machine:

```bash
# Connection and storage settings; override these before calling a helper if
# a machine uses different paths.
export EMU_METEO_LOGIN="ebellini@meteo.ung.si"
export EMU_METEO_JUMP="ebellini@ajdovscina.ung.si"
export EMU_METEO_ROOT="/d4/CAC/ebellini/data/emu_like"
export EMU_LOCAL_DATA_ROOT="/ceph/hpc/home/bellinie/Data_folder"
export HI_FAST_EMU_ROOT="${HOME}/Codes/hi_fast/emu"

emu_sync() {
    local direction="${1:-}"
    local model="${2:-}"
    local subdir="${3:-}"

    if [[ $# -gt 3 || -z "$model" || ( "$direction" != "check" && "$direction" != "push" && "$direction" != "pull" ) ]]; then
        echo "Usage: emu_sync {check|push|pull} MODEL [SUBDIR]"
        return 2
    fi

    # Keep the selection inside the model directory.
    if [[ "$subdir" == /* || "/$subdir/" == */../* ]]; then
        echo "SUBDIR must be a relative path inside MODEL (no .. components)." >&2
        return 2
    fi

    local source_path destination_path push_report pull_report
    local local_path="${EMU_LOCAL_DATA_ROOT:?}/${model}/"
    local meteo_path="${EMU_METEO_LOGIN:?}:${EMU_METEO_ROOT:?}/${model}/"
    local ssh_command="ssh -J ${EMU_METEO_JUMP:?}"
    local local_source="$local_path" meteo_source="$meteo_path"
    local selected_path="$local_path"
    local -a scope_options=()
    if [[ -n "$subdir" ]]; then
        # /./ marks where rsync starts preserving the relative directory tree.
        local_source="${local_path}./${subdir%/}/"
        meteo_source="${meteo_path}./${subdir%/}/"
        selected_path="${local_path}${subdir}/"
        scope_options=(--relative)
    fi

    if [[ "$direction" == "check" ]]; then
        if [[ ! -d "$selected_path" ]]; then
            echo "Local model directory not found: $selected_path" >&2
            return 1
        fi
        push_report=$(rsync -e "$ssh_command" -ahcu \
            --dry-run --itemize-changes \
            "${scope_options[@]}" "$local_source" "$meteo_path") || return
        pull_report=$(rsync -e "$ssh_command" -ahcu \
            --dry-run --itemize-changes \
            "${scope_options[@]}" "$meteo_source" "$local_path") || return

        echo
        echo "Local files newer than or missing from Meteo (would push):"
        if [[ -n "$push_report" ]]; then
            printf '%s\n' "$push_report"
        else
            echo "  None"
        fi
        echo
        echo "Meteo files newer than or missing locally (would pull):"
        if [[ -n "$pull_report" ]]; then
            printf '%s\n' "$pull_report"
        else
            echo "  None"
        fi
        return
    elif [[ "$direction" == "push" ]]; then
        if [[ ! -d "$selected_path" ]]; then
            echo "Local model directory not found: $selected_path" >&2
            return 1
        fi
        source_path="$local_source"
        destination_path="$meteo_path"
    else
        mkdir -p "$local_path" || return
        source_path="$meteo_source"
        destination_path="$local_path"
    fi

    rsync -e "$ssh_command" -avhc --progress --partial \
        "${scope_options[@]}" "$source_path" "$destination_path"
}

hi_fast_sync() {
    local model="$1"

    if [[ -z "$model" ]]; then
        echo "Usage: hi_fast_sync MODEL"
        return 2
    fi

    local local_path="${HI_FAST_EMU_ROOT:?}/${model}/"
    local remote_path="${EMU_METEO_LOGIN:?}:${EMU_METEO_ROOT:?}/${model}/emulator_files/"
    local ssh_command="ssh -J ${EMU_METEO_JUMP:?}"

    mkdir -p "$local_path"
    rsync -e "$ssh_command" -avhc --progress --partial \
        "$remote_path" "$local_path"
}
```

Reload the definitions after editing `~/.bashrc`:

```bash
source ~/.bashrc
```

`emu_sync` is intended for Vega because its local side is Vega's model-data
directory. `hi_fast_sync` is intended for the laptop (or any machine with a
`hi_fast` checkout) because `HI_FAST_EMU_ROOT` controls its local destination.

## 6. Reclaim Vega storage

Only delete the Vega model directory after the final `emu_sync push` has
completed successfully and the Meteo copy has been checked. A useful
read-only comparison is:

```bash
du -sh "${MODEL_DATA}"
ssh -J ebellini@ajdovscina.ung.si ebellini@meteo.ung.si \
    "du -sh '${METEO_MODEL_ROOT}'"
```

Also confirm that `hi_fast_sync` can retrieve the exported files. Deletion is
deliberately not wrapped in a shortcut: verify `MODEL` and `MODEL_DATA`, then
remove the exact model directory manually.

## Quick checklist

- Generate and review sampling YAML/Slurm files.
- Run or resume every sampling job.
- If using range jobs, merge their checkpoints after writers stop.
- Validate all FITS samples and preserve the logs.
- Push the sampled model to Meteo.
- Generate and review training YAML/Slurm files.
- Run or resume every training job.
- Inspect losses and error histograms; preserve the logs.
- Push the trained model to Meteo.
- Export into an empty `emulator_files/` directory.
- Push the export to Meteo.
- From the laptop, pull the export with `hi_fast_sync`.
- Verify both copies before reclaiming Vega storage.
