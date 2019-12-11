from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
from matplotlib.patches import Rectangle
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

class PlansDataset(Dataset):

	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "plans")
		
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'

                # add each example to the dataset (image and xml boundary information)
		for filename in listdir(images_dir):

			image_id = filename[:-4]

			if image_id in ['00090']: # skip bad images # TODO: what are these?
				continue
			# skip all images after 150 if we are building the train set # TODO: why?
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
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

		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	def load_mask(self, image_id):
                """
                Load the masks for an image
                """
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8') # create one array for all masks, each on a different channel

		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('plans'))
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
        raise NotImplementedError("You need to specify the number of classes!")
	NUM_CLASSES = 1 + 1 # Number of classes (background + ?)
	STEPS_PER_EPOCH = 131
        
# Configuration file for evaluation
class PredictionConfig(Config):
	NAME = "plans_cfg"
        raise NotImplementedError("You need to specify the number of classes!")
	NUM_CLASSES = 1 + 1 	# number of classes (background + ?)
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

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
        """
        Plot a number of photos with ground truth and predictions
        """
	for i in range(n_images):
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		scaled_image = mold_image(image, cfg) # convert pixel values (e.g. center)
		sample = expand_dims(scaled_image, 0) # convert image into one sample

		yhat = model.detect(sample, verbose=0)[0]
		pyplot.subplot(n_images, 2, i*2+1)
		pyplot.imshow(image)
		pyplot.title('Actual')
                
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)

		pyplot.subplot(n_images, 2, i*2+2)
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		for box in yhat['rois']:
			y1, x1, y2, x2 = box
			width, height = x2 - x1, y2 - y1
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			ax.add_patch(rect)
	pyplot.show()
