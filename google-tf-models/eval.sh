#!/bin/bash -x

ROOT_DIR=/home/pryldm1/work/google-tf-models
WORK_DIR=/home/pryldm1/work/nexar2/google-tf-models
CONFIG=faster_rcnn_resnet101_01
ID=2017-09-12_${CONFIG}

EXP_DIR=$WORK_DIR/exp/$ID

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim
export CUDA_VISIBLE_DEVICES=0

cd $ROOT_DIR

python3 object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=$WORK_DIR/configs/$CONFIG.config \
        --checkpoint_dir=$EXP_DIR/train \
        --eval_dir=$EXP_DIR/eval
