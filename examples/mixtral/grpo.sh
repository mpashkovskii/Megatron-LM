#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CHECKS_DISABLE=1
export TORCH_NCCL_HIGH_PRIORITY=1

SCRIPTS_DIR=$(dirname $(realpath -s $0))
LOG_DIR=$SCRIPTS_DIR/logs/train/`date +"%Y%m%d_%H%M"`_`uname -n`
mkdir -p $LOG_DIR

DEFAULTS=`grep -o '^[^#]*' /workspace/code/megatron-lm-models/models/defaults/args.txt`
OVERRIDES=(
    # Mixtral 8x7B
    --seq-length 4096
    --num-layers 32
    --max-position-embeddings 32768
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --num-experts 8
    --moe-aux-loss-coeff 1e-3
    --moe-z-loss-coeff 1e-3

    --data-path $DATA_DIR_ROOT/fineweb-edu/sample/10BT/mistralai/Mixtral-8x7B-v0.1/dataset_text_document
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $MODELS_DIR_ROOT/hf/mistralai/Mixtral-8x7B-v0.1/tokenizer.model

    --data-cache-path $DATA_DIR_ROOT/cache
    --num-workers `nproc`
    --no-rope-fusion
    --train-iters 10000
    --tensorboard-dir $LOG_DIR
)

torchrun --nproc_per_node 8 grpo.py \
    ${DEFAULTS[@]} \
    ${OVERRIDES[@]} \
    $@ |& tee $LOG_DIR/output.log
