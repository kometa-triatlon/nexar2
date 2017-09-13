#!/bin/bash -x

ROOT_DIR=/home/pryldm1/work/google-tf-models
WORK_DIR=`pwd`
CONFIG=faster_rcnn_resnet101_01
ID=2017-09-12_${CONFIG}

EXP_DIR=$WORK_DIR/exp/$ID

export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim
export CUDA_VISIBLE_DEVICES=0

ITER=74045
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
