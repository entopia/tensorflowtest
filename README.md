# Mask-RCNN object detection

## Disclosure

This code base was based from the tutorial below, and only briefly tested (do not expect it to run)

## Tutorial
https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

## Mask-RCNN repo
https://github.com/matterport/Mask_RCNN

## Kangaroo dataset
https://github.com/experiencor/kangaroo.git

## Setup

Set up instructions (tested only on MacOS)

```
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
python setup.py install
pip show mask-rcnn
```
If testing the kangaroo dataset, clone the following and put the `kangaroo` folder inside `obj_detection`:
```
git clone https://github.com/experiencor/kangaroo.git
```

And finally,
```
pip install keras
pip install scikit-image
```

### Training the model 

We use transfer learning to train the model; the base model we use is from the mask RCNN model trained on the COCO dataset.
You can download the weights of the model from [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5), and place them inside the `obj_detection` folder.

To continue training the model,
```
python train.py
```

### Evaluate the model

```
python evaluate.py
```
