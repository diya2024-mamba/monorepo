#!/bin/bash

# .env 파일에서 환경 변수 내보내기
set -a
source /data/project/private/chanmuzi/workspace/llm-aia/.env
set +a

# lm-evaluation-harness 폴더로 디렉토리 변경
cd /data/project/private/chanmuzi/workspace/monorepo/projects/llm_eval/lm-evaluation-harness

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
CACHE_DIR=/data/project/llm_engine/.cache/huggingface
OUTPUT_PATH=/data/project/private/chanmuzi/workspace/monorepo/projects/llm_eval/lm-evaluation-harness/outputs/custom_mmlu/1

# 커스텀 MMLU 태스크에 대한 평가 실행
lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,dtype=bfloat16,cache_dir=$CACHE_DIR \
    --tasks mmlu_custom \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path $OUTPUT_PATH \
    --log_samples
