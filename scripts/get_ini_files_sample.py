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
python /ceph/hpc/home/bellinie/emu_like/main.py sample TODO_PATH_YAML -v -r


# ==== END OF JOB COMMANDS ===== #


# Wait for processes, if any.
echo 'Done!'
wait

"""

template_yaml = {
    'output': None,
    'x_sampler': {
        'name': 'latin_hypercube',
        'args': {
            'n_samples': None,
            'seed': 12,
        },      
    },
    'y_model': {
        'name': 'class_spectra',
        'args': None,
        'outputs': None,
    },
    'params': None,
}

parameters_list = {
    'lcdm': ['z_pk', 'h', 'Omega_m', 'Omega_b', 'ln_A_s_1e10', 'n_s', 'tau_reio'],
    'nu': ['N_ur', 'm_ncdm'],
    'k': ['Omega_k'],
}

exclude_parameters = {
    'pk': ['ln_A_s_1e10', 'n_s'],
    'cl': ['z_pk'],
}
    
spectra_list = {
    'pk': [('pk_m', True), ('pk_cb', True), ('pk_weyl', True), ('fk_m', False), ('fk_cb', False), ('fk_weyl', False)],
    'cl': [('cl_TT_lensed', True), ('cl_TE_lensed', False), ('cl_EE_lensed', True), ('cl_pp_lensed', True), ('cl_Tp_lensed', False), ('cl_BB_lensed', True)],
}

parameters_ranges = {
    'z_pk': {
        'thin': [0.,  2.],
        'std':  [0.,  3.],
        'ext':  [0., 10.],
    },
    'h': {
        'thin': [0.65, 0.73],
        'std':  [0.6,  0.8],
        'ext':  [0.5,  0.9],
    },
    'Omega_m': {
        'thin': [0.28, 0.35],
        'std':  [0.2,  0.4],
        'ext':  [0.15, 0.7],
    },
    'Omega_b': {
        'thin': [0.044, 0.047],
        'std':  [0.04,  0.06],
        'ext':  [0.03,  0.07],
    },
    'ln_A_s_1e10': {
        'thin': [2.99, 3.1],
        'std':  [2.9,  3.2],
        'ext':  [2.,   4.],
    },
    'n_s': {
        'thin': [0.95, 0.98],
        'std':  [0.8,  1.1],
        'ext':  [0.7,  1.3],
    },
    'tau_reio': {
        'thin': [0.027, 0.085],
        'std':  [0.02,  0.1],
        'ext':  [0.01,  0.2],
    },
    'N_ur': {
        'thin': [0., 0.4],
        'std':  [0., 1.],
        'ext':  [0., 2.],
    },
    'm_ncdm': {
        'thin': [0., 0.05],
        'std':  [0., 0.12],
        'ext':  [0., 0.4],
    },
    'Omega_k': {
        'thin': [-0.08, -0.03],
        'std':  [-0.1,   0.05],
        'ext':  [-0.2,   0.1],
    },
}

args = {
    'ln_A_s_1e10': 3.044,
    'n_s': 0.966,
    'YHe': 0.24,
    'N_ur': 0.,
    'N_ncdm': 1,
    'deg_ncdm': 3,
    'm_ncdm': 0.02,
    'k_per_decade_for_pk': 40,
    'k_per_decade_for_bao': 80,
    'l_logstep': 1.026,
    'l_linstep': 25,
    'perturbations_sampling_stepsize': 0.02,
    'l_switch_limber': 20,
    'accurate_lensing': 1,
    'delta_l_max': 1000,
    'output': 'tCl, dTk, pCl, lCl, mPk',
    'l_max_scalars': 3000,
    'lensing': 'yes',
    'P_k_max_h/Mpc': 50.0,
    'k_pivot': 0.05,
    'modes': 's',
}


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Settings
    model = 'lcdm'
    n_samples_1000 = 100
    data_root = '/ceph/hpc/data/s25r06-05-users/'

    k_min = 1.e-5
    k_max = 50.
    k_space = 'log'
    k_num = 600
    ell_min = 2
    ell_max = 3000

    ini_folder = '/ceph/hpc/home/bellinie/emu_like/init_files/sample/{}'.format(model)
    io.Folder(ini_folder).create()

    for spectrum in ['pk', 'cl']:
        for parameter_space in ['thin', 'std', 'ext']:


            full_name = 'sample_{}_{}_{}_{}'.format(model, spectrum, n_samples_1000, parameter_space)
            file_name = '{}_{}_{}'.format(spectrum, n_samples_1000, parameter_space)

            # sh file
            with open(os.path.join(ini_folder, 'run_'+file_name+'.sh'), 'w') as fn:
                template_sh_local = template_sh.replace('TODO_NAME', full_name)
                template_sh_local = template_sh_local.replace('TODO_PATH_YAML', os.path.join(ini_folder, file_name+'.yaml'))
                fn.write(template_sh_local)

            # yaml file
            template_yaml['output'] = os.path.join(data_root, '{}/sample/{}_{}_{}.fits'.format(model, spectrum, n_samples_1000, parameter_space))
            template_yaml['x_sampler']['args']['n_samples'] = 1000*n_samples_1000

            # Get list of varied parameters
            parameters = []
            for submodel in parameters_list.keys():
                if submodel in model:
                    for par in parameters_list[submodel]:
                        parameters.append(par)
            # Exclude based on spectra
            for par in exclude_parameters[spectrum]:
                parameters.remove(par)

            # Get dictionary of varied parameters
            template_yaml['params'] = {}
            for par in parameters:
                template_yaml['params'][par] = {
                    'prior': {
                        'min': parameters_ranges[par][parameter_space][0],
                        'max': parameters_ranges[par][parameter_space][1],
                    }
                }
            
            # Get args
            for var in template_yaml['params']:
                try:
                    args.pop(var)
                except KeyError:
                    pass
            template_yaml['y_model']['args'] = args

            # Get outputs
            template_yaml['y_model']['outputs'] = {}
            for sp, ratio in spectra_list[spectrum]:
                template_yaml['y_model']['outputs'][sp] = {}
                if spectrum == 'pk':
                    template_yaml['y_model']['outputs'][sp]['k_min'] = k_min
                    template_yaml['y_model']['outputs'][sp]['k_max'] = k_max
                    template_yaml['y_model']['outputs'][sp]['k_space'] = k_space
                    template_yaml['y_model']['outputs'][sp]['k_num'] = k_num
                if spectrum == 'cl':
                    template_yaml['y_model']['outputs'][sp]['ell_min'] = ell_min
                    template_yaml['y_model']['outputs'][sp]['ell_max'] = ell_max
                if ratio is True:
                    template_yaml['y_model']['outputs'][sp]['ratio'] = True

            with open(os.path.join(ini_folder, file_name+'.yaml'), 'w') as fn:
                yaml.safe_dump(template_yaml, fn, sort_keys=False)

