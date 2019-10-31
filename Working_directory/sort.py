import os
import sys
import io
import re

num_lines = sum(1 for line in open('../annotations/xmls/trainval.txt'))


mypath = '../annotations/xmls/'
newpath = '../annotations/xmls_copy/'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
	f.extend(filenames)
	break




print(f)
newpath_annotations = '/home/minami/Testing/Mask_RCNN/annotations/xmls_copy/'
newpath_images = '/home/minami/Testing/Mask_RCNN/images/queen/'


for foo in range(f.__len__()):
	if f[foo].__contains__("xml"):
		open(str(newpath) + str(f[foo]), 'w').close()
		l = sum(1 for line in open('../annotations/xmls/' + f[foo]))
		with io.open('../annotations/xmls/' + f[foo], 'r') as input:
			with io.open(str(newpath_annotations) + str(f[foo]), 'w') as output:
				a = input.readlines()
				for x in range(a.__len__()):
					result = re.findall("<path>/home/minami/Testing/Mask_RCNN/images/[0-9]+.jpg</path>", a[x])
					if result:
						thisfilename = re.findall("[0-9]+.jpg", result[0])
						a[x] = "        <path>{}{}</path>\n".format(newpath_images, thisfilename[0])
						print(a[x])
				for bar in range(a.__len__()):
					output.write(a[bar])
		input.close()
		output.close()
		print(f[foo])
