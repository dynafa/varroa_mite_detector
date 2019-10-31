import os
import random
import string
import subprocess

def randomString(stringLength=10):
	"""Generate a random string of fixed length """
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for i in range(stringLength))

def main():
	drone = "../classes/drone/images/"
	worker = "../classes/worker/images/"
	queen = "../classes/queen/images/"
	varroa = "../classes/varroa/images/"
	classes = [drone, worker, queen, varroa]
	for x in range(len(classes)):
		i = 1
		# Change all files to random string to prevent being deleted if file name already exists in series 1.jpg, 2.jpg, 3.jpg. etc
		# This is due to merging directories with sequential numbered photos - Queen class only
		for filename in os.listdir(classes[x]):
			dst = randomString(12) + ".jpg"
			src = classes[x] + filename
			dst = classes[x] + dst
			os.rename(src, dst)
			i += 1
		i = 1
		for filename in os.listdir(classes[x]):
			dst = str(i) + ".jpg"
			src = classes[x] + filename
			dst = classes[x] + dst
			os.rename(src, dst)
			i += 1

if __name__ == '__main__':
	subprocess.call("./copy_images.sh")
	main()



