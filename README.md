# nexar2
These are my solutions for the Nexar Challenge 2 (https://www.getnexar.com/challenge-2/).

All the solutions are based on the Faster R-CNN model, but using different frameworks. 
The first one is implemented by Xinlei Chen (https://github.com/endernewton/tf-faster-rcnn),
the second one is the tensorflow/models/object_detection (https://github.com/tensorflow/models/tree/master/research/object_detection).

### tf-faster-rcnn

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

