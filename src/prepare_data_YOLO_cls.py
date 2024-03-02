# Author: Sumukhi Ganesan
# Project: ds5500-capstone-project
# Date: Feb 2023 

# This notebooks converts the CROHME 2023 dataset into Ultralytics YOLO format for Classification

import glob
import os
import shutil
import random
import cv2

def parse_lg_file(filepath):
	    latex_expression = None
	    object_info = []
	    relationships = []
	    bbox = []

	    with open(filepath, 'r') as file:
	    	lines = file.readlines()

	    for line in lines:
	    	if line.startswith('# LaTeX'):
	    		latex_expression = line.split(', ')[1].strip()
	    	elif line.startswith('O'):
	    		obj = line.split(', ')[1:5]
	    		obj[-1] = obj[-1].replace('\n','')
	    		object_info.append(obj)
	    	# elif line.startswith('R'):
	    	# 	rs = line.split(', ')[1:]
	    	# 	rs[-1] = rs[-1].replace('\n','')
	    	# 	relationships.append(rs)
	    	elif line.startswith('BB'):
	    		bb = line.split(', ')[1:6]
	    		bb[-1] = bb[-1].replace('\n','')
	    		bb[1:5] = map(float, bb[1:5])
	    		bbox.append(bb)
	    return latex_expression, object_info, relationships, bbox


def main():

	project_dir = os.path.dirname(os.getcwd())

	data_dir = project_dir + '/datasets'

	input_dir = os.path.dirname(project_dir) + '/data/TC11_CROHME23'

	input_images_dir = input_dir + '/IMG/train/OffHME'
	input_lg_dir = input_dir + '/SymLG/train/OffHME'

	output_dir = data_dir + '/cls_dataset'

	output_train_images_dir = output_dir + '/train'
	output_val_images_dir = output_dir + '/val'
	output_test_images_dir = output_dir + '/test'

	class_count = {}

	all_images = glob.glob(input_images_dir + '/*')

	random.seed(352)

	random.shuffle(all_images)

	train_size = int(len(all_images) * 0.7)
	val_size = int(len(all_images) * 0.15)
	test_size = int(len(all_images) * 0.15)

	# sample_images = [input_images_dir + '/01770.png']

	for i, img_filepath in enumerate(all_images):

		objects = []
		bbox = []
		yolo_boxes = []

		img_filename = img_filepath.split('/')[-1]
		lg_filename = img_filename.replace('.png', '.lg')
		lg_filepath = input_lg_dir + '/' + lg_filename

		img = cv2.imread(img_filepath)
		image_height, image_width, _ = img.shape

		# print(img.shape)

		with open(lg_filepath, 'r') as file:
			for line in file:
				if line.startswith('O'):
					obj = line.split(', ')[1:5]
					obj[-1] = obj[-1].replace('\n','')
					objects.append(obj)
				elif line.startswith('BB'):
					bb = line.split(', ')[1:6]
					bb[-1] = bb[-1].replace('\n','')
					bb[1:5] = map(float, bb[1:5])
					# print(bb)
					bbox.append(bb)

		for j, obj in enumerate(objects):
			symbol_class = obj[1]

			x_min = int(min(bbox[j][1], bbox[j][3]))
			x_max = int(max(bbox[j][1], bbox[j][3]))
			y_min = int(min(bbox[j][2], bbox[j][4]))
			y_max = int(max(bbox[j][2], bbox[j][4]))

			symbol_image = img[y_min:y_max, x_min:x_max]

			if symbol_class not in class_count:
				class_count[symbol_class] = 0
			else:
				class_count[symbol_class] = class_count[symbol_class] + 1

			sym_dest_filename = '{:06d}'.format(class_count[symbol_class]) + '.png'

			if i < train_size:
				sym_dest_path = output_train_images_dir + '/' + symbol_class
			elif i < train_size + val_size:
				sym_dest_path = output_val_images_dir + '/' + symbol_class
			else:
				sym_dest_path = output_test_images_dir + '/' + symbol_class

			# print(sym_dest_path + '/' + sym_dest_filename)

			try:
				if not cv2.imwrite(sym_dest_path + '/' + sym_dest_filename, symbol_image):
					os.mkdir(sym_dest_path)
					cv2.imwrite(sym_dest_path + '/' + sym_dest_filename, symbol_image)
			except:
				print(img_filepath)
				print(img.shape)
				print(obj)
				print(f"{y_min}:{y_max}, {x_min}:{x_max}")

		if i % 100 == 0 :
			print(f"Completed {i} images..")

	print(f"{len(class_count)}  classes identified. The list of classes are:\n")
	print(class_count)


if __name__ == "__main__":
    main()





		
		







