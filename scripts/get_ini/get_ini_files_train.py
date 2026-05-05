import numpy as np
import os
import yaml
import emu_like.io as io

template_sh = """#!/bin/bash

# ---- Metadata configuration ----
#SBATCH --job-name=TODO_NAME
#SBATCH --mail-type=END
#SBATCH --mail-user=emilio.bellini@ung.si


# ---- Resources configuration  ----
#SBATCH --partition=cpu
#SBATCH --mem=62G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/o%j.%x
#SBATCH --error=logs/e%j.%x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32



# ---- Prints  ----
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
echo 'WORKINFO:'
echo 'SLURM: job starting at           '$NOW
echo 'SLURM: sbatch is running on      '$SLURM_SUBMIT_HOST
echo 'SLURM: executing on cluster      '$SLURM_CLUSTER_NAME
echo 'SLURM: executing on partition    '$SLURM_JOB_PARTITION
echo 'SLURM: working directory is      '$SLURM_SUBMIT_DIR
home_dir=$(getent passwd "$SLURM_JOB_ACCOUNT" | cut -d: -f6)
echo "SLURM: current home directory is $home_dir"
echo ""
echo 'JOBINFO:'
echo 'SLURM: job identifier is         '$SLURM_JOBID
echo 'SLURM: job name is               '$SLURM_JOB_NAME
echo ""
echo 'NODEINFO:'
echo 'SLURM: number of nodes is        '$SLURM_JOB_NUM_NODES
echo 'SLURM: number of cpus/node is    '$SLURM_JOB_CPUS_PER_NODE
echo 'SLURM: number of gpus/node is    '$SLURM_GPUS_PER_NODE
echo '------------------------------------------------------'

cd $SLURM_SUBMIT_DIR



# ==== JOB COMMANDS ===== #

module load Python/3.12.3-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0
cd /ceph/hpc/home/bellinie
source ./venv/bin/activate
cd emu_like

#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python /ceph/hpc/home/bellinie/emu_like/main.py train TODO_PATH_YAML -v -f


# ==== END OF JOB COMMANDS ===== #


# Wait for processes, if any.
echo 'Done!'
wait

"""

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


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Settings
    model = 'lcdm'
    timeout = 47
    learning_rate = 1.e-3
    neurons_hidden = [1024, 1024]
    batch_size = 128
    patience = 2000
    reduce_learning_rate = True
    n_samples_1000 = 100
    data_root = '/ceph/hpc/data/s25r06-05-users/'
    time_string = '{:01d}-{:02d}:00:00'.format(*np.divmod(timeout+1, 24))
    num_x_pca = None

    ini_folder = '/ceph/hpc/home/bellinie/emu_like/init_files/train/{}'.format(
        model)
    io.Folder(ini_folder).create()

    for spectrum in spectra_config:
        spectrum_type, num_y_pca, rescale_y = spectra_config[spectrum]

        # Loss function
        if num_y_pca is None:
            loss = 'mean_squared_error'
            loss_floor = None
            loss_delta = None
        else:
            loss = 'mean_squared_error_pca'
            loss_floor = 1.e-4
            loss_delta = 1.e-2

        full_name = 'train_{}_{}'.format(model, spectrum)

        # sh
        with open(os.path.join(ini_folder, 'run_'+spectrum+'.sh'), 'w') as fn:
            template_sh_local = template_sh.replace('TODO_NAME', full_name)
            template_sh_local = template_sh_local.replace(
                'TODO_PATH_YAML', os.path.join(ini_folder, spectrum+'.yaml'))
            template_sh_local = template_sh_local.replace(
                'TODO_TIME', time_string)
            fn.write(template_sh_local)

        # yaml
        template_yaml['output']['path'] = os.path.join(
            data_root, '{}/train/{}/'.format(model, spectrum))
        template_yaml['output']['timeout'] = timeout
        template_yaml['emulator']['args']['learning_rate'] = learning_rate
        template_yaml['emulator']['args']['neurons_hidden'] = neurons_hidden
        template_yaml['emulator']['args']['loss'] = loss
        template_yaml['emulator']['args']['loss_floor'] = loss_floor
        template_yaml['emulator']['args']['loss_delta'] = loss_delta
        template_yaml['emulator']['args']['batch_size'] = batch_size
        template_yaml['emulator']['args']['patience'] = patience
        template_yaml['emulator']['args']['reduce_learning_rate'] = \
            reduce_learning_rate
        template_yaml['datasets']['name'] = spectrum
        template_yaml['datasets']['paths'] = [os.path.join(
            data_root, '{}/sample/{}_{}_{}.fits'.format(
                model, spectrum_type, n_samples_1000, x))
                for x in ['thin', 'std', 'ext']]
        template_yaml['datasets']['rescale_y'] = rescale_y
        template_yaml['datasets']['num_x_pca'] = num_x_pca
        template_yaml['datasets']['num_y_pca'] = num_y_pca

        with open(os.path.join(ini_folder, spectrum+'.yaml'), 'w') as fn:
            yaml.safe_dump(template_yaml, fn, sort_keys=False)
