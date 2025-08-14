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
#SBATCH --mem=30G
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
echo 'SLURM: current home directory is '$(getent passwd $SLURM_JOB_ACCOUNT | cut -d: -f6)
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

cd /ceph/hpc/home/bellinie
source ./venv/bin/activate
cd emu_like

#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python /ceph/hpc/home/bellinie/emu_like/main.py train TODO_PATH_YAML -v


# ==== END OF JOB COMMANDS ===== #


# Wait for processes, if any.
echo 'Done!'
wait

"""

template_yaml = {
    'output': None,
    'emulator': {
        'name': 'ffnn_emu',
        'args': {
            'activation': 'relu',
            'neurons_hidden': [400, 400],
            'batch_normalization': False,
            'dropout_rate': 0.,
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'epochs': 100000,
            'batch_size': 32,
            'patience': 10000,
            'want_output_layer': True,
            'learning_rate': None,
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
        'rescale_x': 'MinMaxScaler',
        'rescale_y': 'MinMaxCommonScaler',
        'num_x_pca': None,
        'num_y_pca': None,
    }
}

spectra_list = {
    'pk': [('pk_m', True), ('pk_cb', True), ('pk_weyl', True), ('fk_m', False), ('fk_cb', False), ('fk_weyl', False)],
    'cl': [('cl_TT_lensed', True), ('cl_TE_lensed', False), ('cl_EE_lensed', True), ('cl_pp_lensed', True), ('cl_Tp_lensed', False), ('cl_BB_lensed', True)],
}


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Settings
    model = 'lcdm'
    learning_rate = 1.e-3
    num_x_pca = None
    num_y_pca = 64

    n_samples_1000 = 100
    data_root = '/ceph/hpc/data/s25r06-05-users/'

    ini_folder = '/ceph/hpc/home/bellinie/emu_like/init_files/train/{}'.format(model)
    io.Folder(ini_folder).create()

    for spectrum_type in ['pk', 'cl']:
        for spectrum, _ in spectra_list[spectrum_type]:

            full_name = 'train_{}_{}_{:.1e}'.format(model, spectrum, learning_rate)
            file_name = '{}_{:.1e}'.format(spectrum, learning_rate)

            # sh
            with open(os.path.join(ini_folder, 'run_'+file_name+'.sh'), 'w') as fn:
                template_sh_local = template_sh.replace('TODO_NAME', full_name)
                template_sh_local = template_sh_local.replace('TODO_PATH_YAML', os.path.join(ini_folder, file_name+'.yaml'))
                fn.write(template_sh_local)

            # yaml
            template_yaml['output'] = os.path.join(data_root, '{}/train/{}/'.format(model, spectrum))
            template_yaml['emulator']['args']['learning_rate'] = learning_rate
            template_yaml['datasets']['name'] = spectrum
            template_yaml['datasets']['paths'] = [
                os.path.join(data_root, '{}/sample/{}_{}_{}.fits'.format(model, spectrum_type, n_samples_1000, x))
                for x in ['thin', 'std', 'ext']]
            template_yaml['datasets']['num_x_pca'] = num_x_pca
            template_yaml['datasets']['num_y_pca'] = num_y_pca

            with open(os.path.join(ini_folder, file_name+'.yaml'), 'w') as fn:
                yaml.safe_dump(template_yaml, fn, sort_keys=False)
