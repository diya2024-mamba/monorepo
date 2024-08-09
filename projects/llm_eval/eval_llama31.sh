#!/bin/bash

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="llama3.1"
selected_subjects="all"
gpu_util=0.8

export CUDA_VISIBLE_DEVICES=0

python eval_mmlupro_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util
