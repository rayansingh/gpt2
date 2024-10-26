#!/bin/bash
#SBATCH --job-name="GPT_CU"
#SBATCH --output="GPT_CU.out"
#SBATCH --error="GPT_CU.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# load model weights for inference
srun chmod u+x ./load_weights.sh
srun ./load_weights.sh

# run kernel output verification
srun ./output_verification

# run model
# srun ./gpt2
