#!/bin/bash

save_dir="projects/llm_eval/eval_results/"
global_record_file="projects/llm_eval/eval_results/eval_record_collection.csv"
model="llama3.1"
selected_subjects="all"
gpu_util=0.8

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python projects/llm_eval/eval_mmlu_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util
