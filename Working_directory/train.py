# fit a mask rcnn on the queen dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import os
import tensorflow as tf
import keras

image_id = 0

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)


# class that defines and loads the queen dataset
class QueenDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define data locations
		images_dir = '../classes/' + dataset_dir + '/images/'
		annotations_dir ='../classes/' + dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 100:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 100:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the filequeen
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('varroa'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class queenConfig(Config):
	# define the name of the configuration
	NAME = "queen_cfg"
	# number of classes (background + classes)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 120
	LEARNING_RATE = 0.0005
	DETECTION_MAX_INSTANCES = 10
	DETECTION_MIN_CONFIDENCE = 0.98
	DETECTION_NMS_THRESHOLD = 0.98

# prepare train set
train_set = QueenDataset()
train_set.add_class("dataset", 1, "varroa")
# train_set.add_class("dataset", 2, "drone")
# train_set.add_class("dataset", 3, "worker")
# train_set.add_class("dataset", 4, "varroa")
# train_set.load_dataset('queen', is_train=True)
# train_set.load_dataset('drone', is_train=True)
# train_set.load_dataset('worker', is_train=True)
train_set.load_dataset('varroa', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = QueenDataset()
test_set.add_class("dataset", 1, "varroa")
# test_set.add_class("dataset", 2, "drone")
# test_set.add_class("dataset", 3, "worker")
# test_set.add_class("dataset", 4, "varroa")
# test_set.load_dataset('queen', is_train=False)
# test_set.load_dataset('drone', is_train=False)
# test_set.load_dataset('worker', is_train=False)
test_set.load_dataset('varroa', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = queenConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
#Declariation of this list is neccessary to prevent errors
model.keras_model.metrics_tensors = []
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')