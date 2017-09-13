#!/usr/bin/python

import os
import argparse
import numpy as np
import pandas as pd
import cv2


colors = {
    'bus' : (255, 255, 255),
    'car' : (255, 0, 0),
    'pickup_truck': (0, 255, 0),
    'truck': (0, 0, 255),
    'van': (0, 255, 255)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--dets', required=True)

    args = parser.parse_args()
    df = pd.read_csv(args.dets)
    for img_id, indices in df.groupby('image_filename').groups.iteritems():
        boxes = []
        img = cv2.imread(os.path.join(args.img_dir, img_id))

        for _, r in df.loc[indices].iterrows():
            if r['confidence'] < 0.75: continue
            x0 = int(r['x0'])
            x1 = int(r['x1'])
            y0 = int(r['y0'])
            y1 = int(r['y1'])
            cv2.rectangle(img, (x0, y0), (x1, y1), color=colors[r['label']], thickness=2)
            cv2.putText(img, '{:s} {:2f}'.format(r['label'], r['confidence']), (x0, y0 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, color=(255, 255, 255))
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
