#!/bin/sh
#SBATCH --job-name=BERT_PyTorch
#SBATCH --output=slurm_out/job_output-%j.txt
#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1

module load cuda/9.0 groupmods/cudnn/9.2

nvidia-smi

N_EXPTS=1
TASKS="Subj"

echo "Starting job at:"
date

# python3 -m pytest -sv ./transformers/tests/ > results/tests.out
# python3 -m pytest -sv ./examples/ > results/examples.out
python3 bert_ex.py > results/bert.out

echo "Job completed at:"
date
