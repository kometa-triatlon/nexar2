# -*- coding: utf-8 -*-
import sys
import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd

import tensorflow as tf
from object_detection.utils import label_map_util

# Print iterations progress
def progressbar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percent, '%', suffix))
    sys.stdout.flush()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def out_detections(outfile, image_id, image_size, category_index, boxes, scores, classes, num, confidence_threshold):
    for i in range(num):
        score = scores[i]
        class_id = classes[i]
        if score < confidence_threshold: continue

        if class_id in category_index.keys():
            class_name = category_index[class_id]['name']
        else:
            class_name = 'unknown'

        ymin, xmin, ymax, xmax = boxes[i, :]
        im_width, im_height = image_size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        outf.write("%s,%.2f,%.2f,%.2f,%.2f,%s,%.4f\n" %( image_id, left, top, right, bottom, class_name, score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to frozen model graph')
    parser.add_argument('--confidence_threshold', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--label_map', required=True, help='Path to label map')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--image_dir', required=True, help='Dir with images')
    parser.add_argument('--index', required=True)
    parser.add_argument('--outfile', required=True)

    args = parser.parse_args()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(args.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args.num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print(category_index)

    index = pd.read_csv(args.index)['image_filename'].tolist()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            outf = open(args.outfile, 'w')
            outf.write('image_filename,x0,y0,x1,y1,label,confidence\n')

            total = len(index)
            progressbar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
            for i, image_id in enumerate(index):
                image_file = os.path.join(args.image_dir, image_id)

                image = Image.open(image_file)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                out_detections(outf,
                               image_id,
                               image.size,
                               category_index,
                               np.squeeze(boxes),
                               np.squeeze(scores),
                               np.squeeze(classes).astype(np.int32),
                               int(num[0]),
                               args.confidence_threshold)

                progressbar(i + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)


