#!/bin/bash -x
set -e

ROOT_DIR=/home/pryldm1/work/google-tf-models
WORK_DIR=`pwd`
CONFIG=faster_rcnn_inception_resnet_v2_02
EXP_DIR=$WORK_DIR/exp/`date +%Y-%m-%d`_$CONFIG

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim
export CUDA_VISIBLE_DEVICES=1

cd $ROOT_DIR
mkdir -p $EXP_DIR/train
mkdir -p $EXP_DIR/eval
mkdir -p $EXP_DIR/log

python3 object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=$WORK_DIR/configs/$CONFIG.config \
        --train_dir=$EXP_DIR/train 2>&1 | tee $EXP_DIR/log/train.log


ITER=`ls -S $EXP_DIR/train/model.ckpt-*.meta | sort -r | grep -oP "(?<=ckpt\-)[0-9]+(?=\.meta)" | head -1`

if [ ! -f $EXP_DIR/frozen_inference_graph.pb ]; then

    cd $ROOT_DIR
    python3 object_detection/export_inference_graph.py \
            --input_type image_tensor \
            --pipeline_config_path $WORK_DIR/configs/$CONFIG.config \
            --trained_checkpoint_prefix $EXP_DIR/train/model.ckpt-$ITER \
            --output_directory $EXP_DIR \
            --optimize_graph

    cd $WORK_DIR
fi

python3 detect.py --model $EXP_DIR/frozen_inference_graph.pb \
        --label_map data/nexet_label_map.pbtxt \
        --num_classes 6 \
        --image_dir data/train \
        --index data/val.csv \
        --outfile $EXP_DIR/nexet_val_dt_${CONFIG}_${ITER}_google-tf.csv
