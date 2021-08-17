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
    python -u -W ignore demo.py \
    --config-file $3 \
    --input ./demo/HICO_test2015_00003124.jpg \
    --opts MODEL.WEIGHTS ./output/hico_det_pretrained.pkl \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log

# python demo.py --config-file ./configs/HICO-DET/interaction_R_50_FPN.yaml \
#   --input ./demo/HICO_test2015_00003124.jpg \
#   --opts MODEL.WEIGHTS ./output/hico_det_pretrained.pkl


# Run demo with pre-trained model (for example, pretrained model on HICO-DET)
# sh ./scripts/demo.sh test 1 ./configs/HICO-DET/interaction_R_50_FPN.yaml
