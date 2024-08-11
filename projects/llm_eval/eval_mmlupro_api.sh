#!/bin/bash

output_dir="projects/llm_eval/eval_results_api/"
model_name="gpt-4o-mini"
assigned_subjects="biology"

export CUDA_VISIBLE_DEVICES=0

python projects/llm_eval/eval_mmlupro_api.py \
                 --output_dir $output_dir \
                 --model_name $model_name \
                 --assigned_subjects $assigned_subjects
