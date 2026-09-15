#!/bin/bash

# ---- Metadata configuration ----
#SBATCH --job-name=inspect_training_lcdm_k
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
jupyter run /ceph/hpc/home/bellinie/emu_like/scripts/inspect_training_lcdm_k.ipynb


# ==== END OF JOB COMMANDS ===== #


# Wait for processes, if any.
echo 'Done!'
wait

