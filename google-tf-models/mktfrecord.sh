#!/bin/bash -x

ROOT_DIR=/home/pryldm1/work/google-tf-models/research
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/slim


for split in val train; do
python3 create_nexar_tfrecord.py --logtostderr \
        --img_dir data/train \
        --gt_file data/train_boxes.csv \
        --index data/$split.csv \
        --label_map_path nexet_label_map.pbtx \
        --output_file data/$split.tfrecord
done
