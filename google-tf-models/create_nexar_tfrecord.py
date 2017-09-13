#!/usr/bin/env python3
import hashlib
import io
import logging
import os
import random
import pandas as pd

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('img_dir', '', 'Directory with images')
flags.DEFINE_string('gt_file', 'train_boxes.csv', 'File with ground truth annotations')
flags.DEFINE_string('index', 'train.csv', 'Index file of the set')
flags.DEFINE_string('label_map_path', 'nexet_label_map.pbtx', 'Path to label map proto')
flags.DEFINE_string('output_file', '', 'Path to the output TFRecords file')
FLAGS = flags.FLAGS

class Box:
  def __init__(self, x0, y0, x1, y1, label, confidence):
    self.xmin = x0
    self.ymin = y0
    self.xmax = x1
    self.ymax = y1
    self.label = label
    self.confidence = confidence

def make_tf_example(img_name, boxes, label_map_dict, img_dir):
  img_path = os.path.join(img_dir, img_name)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  width, height = image.size 

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  for box in boxes:
    xmin.append(box.xmin/width)
    ymin.append(box.ymin/height)
    xmax.append(box.xmax/width)
    ymax.append(box.ymax/height)
    
    class_name = box.label
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          img_name.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          img_name.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def main(_):
  img_dir = FLAGS.img_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from Nexet dataset.')

  writer = tf.python_io.TFRecordWriter(FLAGS.output_file)

  gt_df = pd.read_csv(FLAGS.gt_file)
  index = pd.read_csv(FLAGS.index)['image_filename'].tolist()

  logging.info('%d images in the index.', len(index))

  df = gt_df[gt_df['image_filename'].isin(index)]
  for img_name, indices in df.groupby('image_filename').groups.items():
    boxes = [Box(r['x0'], r['y0'], r['x1'], r['y1'], r['label'], r['confidence']) for _, r in gt_df.loc[indices].iterrows()]
    tf_example = make_tf_example(img_name, boxes, label_map_dict, img_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()
  
  
if __name__ == '__main__':
  tf.app.run()
