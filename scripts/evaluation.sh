#!/bin/sh
# @Author: Daoming Zong
# @Date: 2021-09-03 02:18:07
# @LastEditTime: 2021-09-07 17:43:01
# @LastEditors: Daoming Zong
# @Description: 
# @FilePath: /models/HOTR/scripts/evaluation.sh
# Copyright (c) 2021 SenseTime IRDC Group. All Rights Reserved.

set -x 

PARTITION=$1
GPUS=$2
config=$3

declare -u expname
expname=`basename ${config} .yaml`

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

currenttime=`date "+%Y%m%d%H%M%S"`
g=$(($2<8?$2:8))

mkdir -p  results/${expname}/train_log

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${expname} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=$g \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -W ignore main.py \
    --config $3 \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log
    
# sh ./scripts/evaluation.sh test 8 ./configs/eval_vcoco.yaml

# sh ./scripts/evaluation.sh test 8 ./configs/eval.yaml

