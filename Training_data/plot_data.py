import matplotlib.pyplot as plt
import numpy as np
import io
import re

data_array = np.zeros([10], dtype=np.float16)
data_array2 = np.zeros([10], dtype=np.float16)
print(data_array)

fig = plt.figure()
ax = plt.subplot(111)

with io.open('./data2.txt', 'r') as data:
	pos = 0
	line = data.readline()
	while line:
		match = re.findall("loss: [0-9].[0-9]+", line)
		# match2 = re.findall("val-loss: 0.[0-9]+", line)
		if match:
			val = np.float16(match[0][6:])
			print(type(val))
			data_array[pos] = val
			print(val)
			pos += 1
		# if match2:
		# 	val = np.float16(match2[0][10:])
		# 	print(type(val))
		# 	data_array2[pos] = val
		# 	print(val)
		line = data.readline()
data.close()

with io.open('./data2.txt', 'r') as data:
	pos = 0
	line = data.readline()
	while line:
		match = re.findall("val_loss: [0-9].[0-9]+", line)
		# match2 = re.findall("val-loss: 0.[0-9]+", line)
		print(match)
		if match:
			val = np.float16(match[0][9:])
			print(type(val))
			data_array2[pos] = val
			print(val)
			pos += 1
		# if match2:
		# 	val = np.float16(match2[0][10:])
		# 	print(type(val))
		# 	data_array2[pos] = val
		# 	print(val)
		line = data.readline()
data.close()


print(data_array)
print(data_array2)
ax.plot(data_array, label='Loss')
ax.plot(data_array2, label='Val_loss')
ax.legend()
plt.plot(data_array)
plt.plot(data_array2)
plt.title("Training loss - 4 classes, 10 epochs, LR=0.001")
# plt.plot(data_array2)
plt.savefig("../../DeepLearning/Assignment/images/training_results2.png", dpi=100)