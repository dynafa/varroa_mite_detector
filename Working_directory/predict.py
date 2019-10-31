# detect queens in photos with mask rcnn model
from os import listdir, environ
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "0"

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

pyplot.subplots_adjust(left=0.05)
pyplot.subplots_adjust(right=0.95)
pyplot.subplots_adjust(top=0.95)
pyplot.subplots_adjust(bottom=0.05)
pyplot.subplots_adjust(wspace=0.05)
pyplot.subplots_adjust(hspace=0.05)

# class that defines and loads the queen dataset
class QueenDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
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
			if is_train and int(image_id) >= 120:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 120:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
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
			# class_ids.append(self.class_names.index('queen'))
			# class_ids.append(self.class_names.index('drone'))
			# class_ids.append(self.class_names.index('worker'))
			class_ids.append(self.class_names.index('varroa'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "queen_cfg"
	# number of classes (background + queen)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MAX_INSTANCES = 10
	DETECTION_MIN_CONFIDENCE = 0.99
	DETECTION_NMS_THRESHOLD = 0.99

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, index):
	# load image and mask
	# for i in range(n_images):
		# load the image and mask
	name = str(index) + ".jpg"
	image = dataset.load_image(index)
	mask, _ = dataset.load_mask(index)
	# convert pixel values (e.g. center)
	scaled_image = mold_image(image, cfg)
	# convert image into one sample
	sample = expand_dims(scaled_image, 0)
	# make prediction
	yhat = model.detect(sample, verbose=1)[0]
	# define subplot
	pyplot.subplot(2, 1, 1)
	# plot raw pixel data
	pyplot.imshow(image)
	pyplot.title('Actual')
	# plot masks
	for j in range(mask.shape[2]):
		pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
	# get the context for drawing boxes
	pyplot.subplot(2, 1, 2)
	# plot raw pixel data
	pyplot.imshow(image)
	pyplot.title('Predicted')
	ax = pyplot.gca()
	# plot each box
	for box in yhat['rois']:

		print(yhat['class_ids'])
		print(yhat['scores'])
		# get coordinates
		y1, x1, y2, x2 = box
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		pyplot.text(x1, y1, yhat['class_ids'][0], color='white')
		pyplot.text(x1, y1 + 5, yhat['scores'][0], color='white')

	# show the figure
	pyplot.savefig(name)

# load the train dataset
train_set = QueenDataset()
train_set.add_class("dataset", 1, "varroa")
# train_set.add_class("dataset", 2, "drone")
# train_set.add_class("dataset", 3, "worker")
# train_set.add_class("dataset", 4, "varroa")
# # train_set.load_dataset('queen', is_train=True)
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


# sys.exit(0)

print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
cfg.display()

# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'queen_cfg20191030T1442/mask_rcnn_queen_cfg_0030.h5'
model.load_weights(model_path, by_name=True)

for x in range(50):
	# plot predictions for train dataset
	plot_actual_vs_predicted(train_set, model, cfg, x)
	# plot predictions for test dataset
	# plot_actual_vs_predicted(test_set, model, cfg, x)