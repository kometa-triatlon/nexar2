# nexar2
These are my solutions for the Nexar Challenge 2 (https://www.getnexar.com/challenge-2/).

All the solutions are based on the Faster R-CNN model, but using different frameworks. 
The first one is implemented by Xinlei Chen (https://github.com/endernewton/tf-faster-rcnn),
the second one is the tensorflow/models/object_detection (https://github.com/tensorflow/models/tree/master/research/object_detection).

### tf-faster-rcnn

The following instructions must be performed under the `tf-faster-rcnn` directory.

**1. Framework setup**

   The setup process described in https://github.com/endernewton/tf-faster-rcnn.

**2. Data preparation**

   Create a symlink to the nexet dataset inside the `data` folder:
```Shell
ln -s /path/to/nexet data/nexet
```
   The nexet folder must contain index files (`train.csv`, `val.csv`, `test.csv`) and the folder with images.

**3. Training**

   To start the training, run:
```Shell
experiments/scripts/train_faster_rcnn.sh 0 res101
```

   The resulting model will be stored under `output/res101/nexet_train/default`.

**4. Producing detection outputs**

```Shell
cd experiments
cp -r eval_template eval_faster_rcnn_res101
cd eval_faster_rcnn_res101
./run.sh
```
This will produce the output file named like `nexet_val_dt_res101_nexet_train_465000.csv`.
If you want to run the model on the test set, adjust the `IMGDIR` and `NDX` variables accordingly. 

**Experimental results**

The Faster R-CNN based on ResNet-101 and trained only on the Nexet dataset achieves 0.7888 mAP on the validation set.


### google tensorflow-models

The following instructions must be performed under the `google-tf-models` directory.

**1. Framework setup**

The setup process described here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

You must have TensorFlow installed and running. Also, you must have the models repository cloned locally.
I created a symlink to it inside the `google-tf-models`:

```Shell
ln -s /path/to/tensorflow/models/research/object_detection object_detection
```

**2. Data preparation**

Before training, you must convert your data into TFRecords format:

```Shell
ln -s /path/to/nexet data
./mktfrecord.sh

```

**Note:** Set the `ROOT_DIR` variable to point at your local `tensorflow/models/research` repo.


**3. Training**

First, download the pre-trained weights, required for the initial checkpoint. 

Depending on the config you are using, these are the following:

 - `faster_rcnn_resnet101_02.config` requires http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz to be stored at `object_detection/models/resnet_v1_101.ckpt`.
 - `faster_rcnn_inception_resnet_v2_01.config` requires http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz to be unpacked under `object_detection/models/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017`.

 - `faster_rcnn_inception_resnet_v2_02.config` requires http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz to be stored at `object_detection/models/inception_resnet_v2_2016_08_30.ckpt`.


Before starting the training, edit the `train.sh` to adjust the `ROOT_DIR` and to specify the config you want to use. Also, you might want to change or to remove the ` CUDA_VISIBLE_DEVICES` variable to use the GPU you want.

Then:
```Shell
./train.sh
```

**4. Producing detection outputs**

```Shell
./detect.sh
```

**Note:** Before running the script, adjust the `ROOT_DIR`, `CONFIG`, `ID`, and `ITER` variables to pick the model you want to evaluate. Also, choose the value for the `FOLD` variable between `val` and `test`.

**Experimental results**

| **Config** | **Iter** | **Fine-tuned from** | **Validation mAP** | **Test mAP** |
|:------:|:----:|:---------------:|:--------------:|:--------:|
| faster_rcnn_resnet101_02| 500k | ResNet V2 101 trained on ImageNet | 0.7708 | - |
| faster_rcnn_inception_resnet_v2_01 | 183k | Faster R-CNN Inception ResNet V2 trained on MSCOCO | 0.8121 | 0.7564 |
| faster_rcnn_inception_resnet_v2_02 | 186k | Inception ResNet V2 trained on ImageNet | 0.8028 | 0.7468 |

