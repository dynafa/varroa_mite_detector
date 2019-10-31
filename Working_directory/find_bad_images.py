import os

def main():
	drone = "/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/drone/images/"
	worker = "/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/worker/images/"
	queen = "/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/queen/images/"
	varroa = "/home/kamoto/Workspaces/Workspace_1/EIT2019/ML/Testing/Mask_RCNN/classes/varroa/images/"
	classes = [drone, worker, queen, varroa]
	total = 0
	for x in range(len(classes)):
		i = 1
		j = 0
		print(classes[x])
		os.chdir(classes[x])
		for filename in os.listdir(classes[x]):
			if filename.__contains__(".jpg"):
				image_id = filename[:-4]
				if not os.path.exists(image_id + ".xml"):
					print(filename, image_id)
					j += 1
					total += 1
				i += 1
		print(j)
	print(total)

if __name__ == '__main__':
	main()
