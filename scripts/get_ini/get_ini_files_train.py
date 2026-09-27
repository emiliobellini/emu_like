"""Generate standard or Sobolev training configurations and Slurm launchers."""
import argparse
from copy import deepcopy
from pathlib import Path
import shlex

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

template_yaml = {
    'output': {
        'path': None,
        'timeout': None,
    },
    'emulator': {
        'name': 'ffnn_emu',
        'args': {
            'activation': 'relu',
            'neurons_hidden': None,
            'batch_normalization': False,
            'dropout_rate': 0.,
            'optimizer': 'adam',
            'loss': None,
            'loss_floor': None,
            'loss_delta': None,
            'epochs': 100000,
            'batch_size': None,
            'patience': None,
            'want_output_layer': True,
            'learning_rate': None,
            'reduce_learning_rate': None,
            'relative_improvement': None,
        },
    },
    'datasets': {
        'paths': None,
        'paths_x': None,
        'paths_y': None,
        'columns_x': None,
        'columns_y': None,
        'name': None,
        'remove_non_finite': True,
        'frac_train': 0.9,
        'train_test_random_seed': 1543,
        'rescale_x': 'StandardScaler',
        'rescale_y': None,
        'num_x_pca': None,
        'num_y_pca': None,
    }
}

spectra_config = {
    #                type, pca,  y_scaler,
    'pk_m':         ('pk', None, 'LogStandardScaler'),
    'pk_cb':        ('pk', None, 'LogStandardScaler'),
    'pk_weyl':      ('pk', None, 'LogStandardScaler'),
    'fk_m':         ('pk', None, 'StandardScaler'),
    'fk_cb':        ('pk', None, 'StandardScaler'),
    'fk_weyl':      ('pk', None, 'StandardScaler'),
    'cl_TT_lensed': ('cl', 360,  'LogStandardScaler'),
    'cl_TE_lensed': ('cl', 360,  'StandardScaler'),
    'cl_EE_lensed': ('cl', 360,  'LogStandardScaler'),
    'cl_pp_lensed': ('cl', 360,  'LogStandardScaler'),
    'cl_Tp_lensed': ('cl', 360,  'StandardScaler'),
    'cl_BB_lensed': ('cl', 360,  'LogStandardScaler'),
}


def generate(model='lcdm', sobolev=False, GPU=False, *,
             repo_root=REPO_ROOT, data_root='/ceph/hpc/data/s25r06-05-users',
             timeout=47):
    """Write one YAML and launcher per spectrum; GPU variants use _gpu names.

    Sobolev generates only pk_* networks, each learning its paired fk_* target.
    CPU and GPU variants share the same training output directory.
    """
    repo_root = Path(repo_root).resolve()
    ini_folder = repo_root / 'init_files' / 'train' / model
    output_root = Path(data_root) / model / 'train'
    if sobolev:
        ini_folder /= 'sobolev'
        output_root /= 'sobolev'
    ini_folder.mkdir(parents=True, exist_ok=True)
    (repo_root / 'logs').mkdir(exist_ok=True)
    days, hours = divmod(timeout + 1, 24)
    time_string = f'{days}-{hours:02d}:00:00'
    generated = []

    for spectrum, config in spectra_config.items():
        spectrum_type, num_y_pca, rescale_y = config
        if sobolev and not spectrum.startswith('pk_'):
            continue
        params = deepcopy(template_yaml)
        params['output'].update(path=str(output_root / spectrum) + '/',
                                timeout=timeout)
        args = params['emulator']['args']
        args.update(
            learning_rate=1.e-3, neurons_hidden=[1024, 1024],
            loss=('mean_squared_error_pca' if num_y_pca
                  else 'mean_squared_error'),
            loss_floor=1.e-4 if num_y_pca else None,
            loss_delta=1.e-2 if num_y_pca else None,
            batch_size=512 if sobolev else 128, patience=2000,
            reduce_learning_rate=True, relative_improvement=True)
        params['datasets'].update(
            name=spectrum,
            paths=[str(Path(data_root) / model / 'sample' /
                       f'{spectrum_type}_100_{region}.fits')
                   for region in ['thin', 'std', 'ext']],
            rescale_y=rescale_y, num_x_pca=None, num_y_pca=num_y_pca)
        if sobolev:
            params['emulator']['name'] = 'sobolev_ffnn_emu'
            for key in ('loss', 'loss_floor', 'loss_delta'):
                del args[key]
            args.update(activation='tanh', pk_weight=1.0, fk_weight=1.0,
                        fk_loss='standardized_mse', fk_warmup_epochs=2000,
                        fk_ramp_epochs=2000)
            params['datasets']['rescale_growth'] = 'StandardScaler'

        stem = spectrum + ('_gpu' if GPU else '')
        yaml_path = ini_folder / (stem + '.yaml')
        yaml_path.write_text(yaml.safe_dump(params, sort_keys=False))
        job_name = f'train_{model}_' + ('sobolev_' if sobolev else '') + stem
        mem = '62G' if sobolev or spectrum_type == 'cl' else '24G'
        gpu_resources = '#SBATCH --gres=gpu:1\n' if GPU else ''
        gpu_setup = """module load CUDA/12.6.0
module load cuDNN/9.10.2.21-CUDA-12.6.0
nvidia-smi
python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow GPUs:', gpus)
assert gpus, 'No GPU visible to TensorFlow'
PY
""" if GPU else ''
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END
#SBATCH --mail-user=emilio.bellini@ung.si
#SBATCH --partition={'gpu' if GPU else 'cpu'}
{gpu_resources}#SBATCH --mem={mem}
#SBATCH --time={time_string}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={16 if GPU else 32}
#SBATCH --output={repo_root}/logs/o%j.%x
#SBATCH --error={repo_root}/logs/e%j.%x

set -euo pipefail
REPO_ROOT={shlex.quote(str(repo_root))}
cd "$REPO_ROOT"
module load Python/3.12.3-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0
source /ceph/hpc/home/bellinie/venv/bin/activate
export PYTHONPATH="$REPO_ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
{gpu_setup}
# Resume strictly when output exists; otherwise start a new training.
srun --cpu-bind=cores python "$REPO_ROOT/main.py" train \\
  {shlex.quote(str(yaml_path))} -v -f -r "$@"
"""
        sh_path = ini_folder / ('run_' + stem + '.sh')
        sh_path.write_text(script)
        sh_path.chmod(sh_path.stat().st_mode | 0o111)
        generated.append((yaml_path, sh_path))
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='lcdm')
    parser.add_argument('--sobolev', action=argparse.BooleanOptionalAction,
                        default=False, help='Generate paired Pk/fk training')
    parser.add_argument('--GPU', '--gpu', dest='GPU',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Generate GPU launchers and *_gpu.yaml files')
    cli = parser.parse_args()
    generate(model=cli.model, sobolev=cli.sobolev, GPU=cli.GPU)
