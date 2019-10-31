# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree
from os import listdir
from os.path import isfile, join
from sys import stdout

mypath = '../annotations/xmls/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
	f.extend(filenames)
	break


# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
	# load and parse the file
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


# extract details form annotation file
# summarize extracted details

for foo in range(f.__len__()):
	if f[foo].__contains__("xml"):
		boxes, w, h = extract_boxes('../annotations/xmls/'+f[foo])
		stdout.write(f[foo])
		stdout.flush()
		print(boxes, w, h)