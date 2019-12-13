import time
import os
from os import listdir
from os import path
from xml.etree import ElementTree
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
import matplotlib.patches
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


"""

This file contains classes and helper functions used to train and evaluate
a keras objection detection model using transfer learning.

IMPORTANT!
The code in this file is largely refactored from the following tutorial:
https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

"""

COLOURS = {0:"black", 1:"magenta", 2:"green", 3:"steelblue"}
CLASS_ID_TO_NAME = {0:"background", 1:"drawing", 2:"drawingtitle", 3:"blocktitle"}

class PlansDataset(Dataset):

	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "drawing")
		self.add_class("dataset", 2, "drawingtitle")
		self.add_class("dataset", 3, "blocktitle")

		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'

		num_examples = len(listdir(images_dir))
		if is_train: # TODO: specify training size
			data_files = listdir(images_dir)[:int(num_examples*0.8)]
		else:
			data_files = listdir(images_dir)[int(num_examples*0.8):]

		# add each example to the dataset (image and xml boundary information)
		for filename in data_files:

			image_id = path.splitext(filename)[0]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	def extract_boxes(self, filename):
		"""
		Extract bounding boxes from an annotation file
		"""
		tree = ElementTree.parse(filename)
		root = tree.getroot()
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)

		names = list()
		for name in root.findall('.//object'):
			n = name.find('name').text
			names.append(n)

		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height, names

	def load_mask(self, image_id):
		"""
		Load the masks for an image
		"""
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h, names = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8') # create one array for all masks, each on a different channel

		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(names[i]))
		return masks, asarray(class_ids, dtype='int32')

	def image_reference(self, image_id):
		"""
		Load an image reference
		"""
		info = self.image_info[image_id]
		return info['path']

# Configuration file for analysis
class PlansConfig(Config):
	NAME = "plans_cfg"
	NUM_CLASSES = 3 + 1 # Number of classes (background + ?)
	STEPS_PER_EPOCH = 10
        
# Configuration file for evaluation
class PredictionConfig(Config):
	NAME = "plans_cfg"
	NUM_CLASSES = 3 + 1 	# number of classes (background + ?)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
	"""
	Calculate the mAP for a model on a given dataset
	"""
	APs = list()
	for image_id in dataset.image_ids:

		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		scaled_image = mold_image(image, cfg) # convert pixel values (e.g. center)
		sample = expand_dims(scaled_image, 0) # convert image into one sample
		
		yhat = model.detect(sample, verbose=0) # make prediction
		r = yhat[0] # extract results for first sample
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		APs.append(AP)
		
	mAP = mean(APs)
	return mAP

def plot_actual_vs_predicted(dataset, model, cfg, n_images):
	"""
	Plot a number of photos with ground truth and predictions
	"""
	for j in range(n_images):

                image = dataset.load_image(j)
                mask, _ = dataset.load_mask(j)
                scaled_image = mold_image(image, cfg) # convert pixel values (e.g. center)
                sample = expand_dims(scaled_image, 0) # convert image into one sample
                
                fig = pyplot.figure(figsize=(20,20))
                yhat = model.detect(sample, verbose=0)[0]
                pyplot.subplot(1, 2, 1)
                pyplot.imshow(image)
                pyplot.title('Actual', fontsize=24)
                pyplot.axis('off')
                pyplot.imshow(np.any(mask, axis=2), cmap='gray', alpha=0.3)
                pyplot.subplot(1, 2, 2)
                pyplot.imshow(image)
                pyplot.title('Predicted', fontsize=24)
                ax = pyplot.gca()
                for i, box in enumerate(yhat['rois']):
                        y1, x1, y2, x2 = box
                        width, height = x2 - x1, y2 - y1
                        rect = matplotlib.patches.Rectangle((x1, y1), width, height, fill=False, color=COLOURS[yhat["class_ids"][i]])
                        ax.add_patch(rect)
                
                pyplot.tight_layout()
                pyplot.axis('off')
                handles = [matplotlib.patches.Patch(color=COLOURS[i], label=CLASS_ID_TO_NAME[i]) for i in COLOURS.keys()]
                pyplot.legend(handles=handles, loc="lower right", prop={'size': 18})

                os.makedirs("predicted_images", exist_ok=True)
                pyplot.savefig("predicted_images/{}_{}.png".format(j, time.time()), dpi=500, bbox_inches = 'tight')
