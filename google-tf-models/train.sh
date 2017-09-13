#!/bin/bash -x

ROOT_DIR=/home/pryldm1/work/google-tf-models
WORK_DIR=/home/pryldm1/work/nexar2/google-tf-models
CONFIG=faster_rcnn_resnet101_01
EXP_DIR=$WORK_DIR/exp/`date +%Y-%m-%d`_$CONFIG

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim
export CUDA_VISIBLE_DEVICES=1

cd $ROOT_DIR
mkdir -p $EXP_DIR/train
mkdir -p $EXP_DIR/eval

python3 object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=$WORK_DIR/configs/$CONFIG.config \
        --train_dir=$EXP_DIR/train

#python3 object_detection/eval.py \
#        --logtostderr \
#        --pipeline_config_path=$WORK_DIR/configs/$CONFIG.config \
#        --checkpoint_dir=$EXP_DIR/train \
#        --eval_dir=$EXP_DIR/eval
