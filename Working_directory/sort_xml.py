import os
import sys
import io
import re

from os import listdir
from os.path import isfile, join
from os import walk


queenpath = '../classes/queen/annots/'
workerpath = '../classes/worker/annots/'
varroapath = '../classes/varroa/annots/'
dronepath = '../classes/drone/annots/'

newqueenpath = '../classes/queen/xmls_copy/'
newworkerpath = '../classes/worker/xmls_copy/'
newvarroapath = '../classes/varroa/xmls_copy/'
newdronepath = '../classes/drone/xmls_copy/'

queenpathstring = "<path>/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/queen/images/[0-9]+.jpg</path>"
workerpathstring = "<path>/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/worker/images/[0-9]+.jpg</path>"
varroapathstring = "<path>/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/varroa/images/[0-9]+.jpg</path>"
dronepathstring = "<path>/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/drone/images/[0-9]+.jpg</path>"


newqueenpathstring = "/home/minami/Testing/Mask_RCNN/classes/queen/images/"
newworkerpathstring = "/home/minami/Testing/Mask_RCNN/classes/worker/images/"
newvarroapathstring = "/home/minami/Testing/Mask_RCNN/classes/varroa/images/"
newdronepathstring = "/home/minami/Testing/Mask_RCNN/classes/drone/images/"

classes = [queenpath, workerpath, varroapath, dronepath]
newclasses = [newqueenpath, newworkerpath, newvarroapath, newdronepath]
pathstring = [queenpathstring, workerpathstring, varroapathstring, dronepathstring]
newpathstring = [newqueenpathstring, newworkerpathstring, newvarroapathstring, newdronepathstring]

for foobar in range(len(classes)):
	onlyfiles = [f for f in listdir(classes[foobar]) if isfile(join(classes[foobar], f))]
	f = []
	for (dirpath, dirnames, filenames) in walk(classes[foobar]):
		f.extend(filenames)
		break
	for foo in range(f.__len__()):
		if f[foo].__contains__("xml"):
			open(str(newclasses[foobar]) + str(f[foo]), 'w').close()
			with io.open(str(classes[foobar]) + str(f[foo]), 'r') as input:
				with io.open(str(newclasses[foobar]) + str(f[foo]), 'w') as output:
					a = input.readlines()
					for x in range(a.__len__()):
						result = re.findall(pathstring[foobar], a[x])
						if result:
							thisfilename = re.findall("[0-9]+.jpg", result[0])
							a[x] = "        <path>{}{}</path>\n".format(newpathstring[foobar], thisfilename[0])
							print(a[x])
					for bar in range(a.__len__()):
						output.write(a[bar])
			input.close()
			output.close()
