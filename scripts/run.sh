 #!/bin/sh
# GPUS_PER_NODE=16

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


# export DETECTRON2_DATASETS='/mnt/lustre/share_data/zongdaoming/datasets/'

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${expname} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=$g \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -W ignore train_net.py \
    --config-file $3 \
    --num-machines 1 \
    --num-gpus $g \
    --dist-url auto \
    # --eval-only \
    OUTPUT_DIR "./output/HICO_interaction_zero_shot" \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log

# cd tools/
# ./train_net.py --num-gpus 8 \
#   --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
# ./train_net.py \
#   --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
# ./train_net.py \
#   --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#   --eval-only MODEL.WEIGHTS /path/to/checkpoint_file

# The following examples train a model to detect human interactions with novel objects
# using hico-det_train set. The only difference from the above is to use a class agnostic bbox regressor here.


# sh ./scripts/run.sh test 4 ./configs/HICO-DET/interaction_zero_shot_R_50_FPN.yaml