#!/bin/bash

output_dir="./data/eval_results_api/"
model_name="gpt-4o-mini"
assigned_subjects="biology"
batch_size=8
# export CUDA_VISIBLE_DEVICES=0

python ./eval_mmlupro_api.py \
                 --output_dir $output_dir \
                 --model_name $model_name \
                 --assigned_subjects $assigned_subjects \
                 --batch_size $batch_size
