#!/bin/sh
###
 # @Author: your name
 # @Date: 2021-08-26 15:21:45
 # @LastEditTime: 2021-09-07 21:48:13
 # @LastEditors: Daoming Zong and Chunya Liu
 # @Description: In User Settings Edit
 # @FilePath: /models/HOTR/scripts/train.sh
### 
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
    

# sh ./scripts/train.sh irdcRD 16 ./configs/hotr_vcoco.yaml
# sh ./scripts/train.sh irdcRD 16 ./configs/hotr_resnet.yaml
# sh ./scripts/train.sh irdcRD 16 ./configs/hotr_vcoco_resnet.yaml
# sh ./scripts/train.sh test 1 ./configs/hotr_vcoco_resnet.yaml
# sh ./scripts/train.sh test 8 ./configs/hotr.yaml


# sh ./scripts/train.sh irdcRD 16 ./configs/hotr.yaml


# sh ./scripts/train.sh irdcRD 24 ./configs/zeroshot/zero_shot_rand_interactiveness.yaml

# sh ./scripts/train.sh irdcRD 24 ./configs/zeroshot/zero_shot_rand_interactiveness.yaml



# sh ./scripts/train.sh test 8 ./configs/zeroshot/zero_shot_rand_interactiveness_frozen_false.yaml