#!/bin/bash -l
#SBATCH -J simpleDemoCompareCost
#SBATCH -o diag/%x_output_%A.txt
#SBATCH -e diag/%x_errors_%A.txt
#SBATCH -p serial
#SBATCH --array 1-100
#SBATCH -t 02:00:00
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=4000

module load julia-env
srun julia "${SLURM_JOB_NAME}.jl" \
  1 \
  "output/${SLURM_JOB_NAME}_results_${SLURM_ARRAY_TASK_ID}.jld2" \
  "${SLURM_ARRAY_TASK_ID}"
