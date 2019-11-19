#!/bin/sh
#SBATCH --job-name=BERT_CNN_PyTorch
#SBATCH --output=slurm_out/job_output-%j.txt
#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:titanrtx:1

module load cuda/9.0 groupmods/cudnn/9.2

nvidia-smi
# export TASK_NAME="CoLA"
export TASK_NAME="MRPC"

echo "Starting job at:"
date

time python3 simpletran_bert.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --overwrite_output_dir \
    --output_dir $BERT_OUT/$TASK_NAME/ > results/bert_cnn_ft.out

echo "Job completed at:"
date
