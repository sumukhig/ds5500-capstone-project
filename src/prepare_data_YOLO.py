# Author: Sumukhi Ganesan
# Project: ds5500-capstone-project
# Date: Feb 2023 

# This notebooks converts the CROHME 2023 dataset into Ultralytics YOLO format

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

# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo_format(bbox, width, height):
	x_min = min(bbox[0], bbox[2])
	x_max = max(bbox[0], bbox[2])
	y_min = min(bbox[1], bbox[3])
	y_max = max(bbox[1], bbox[3])
	x_center = (x_min + x_max) / (2 * width)
	y_center = (y_min + y_max) / (2 * height)
	bbox_width = (x_max - x_min) / width
	bbox_height = (y_max - y_min) / height
	return x_center, y_center, bbox_width, bbox_height

def main():

	project_dir = os.path.dirname(os.getcwd())

	data_dir = project_dir + '/datasets'

	input_dir = os.path.dirname(project_dir) + '/data/TC11_CROHME23'

	input_images_dir = input_dir + '/IMG/train/OffHME'
	input_lg_dir = input_dir + '/SymLG/train/OffHME'

	output_dir = data_dir + '/custom_dataset'

	output_images_dir = output_dir + '/images'
	output_labels_dir = output_dir + '/labels'

	output_train_images_dir = output_images_dir + '/train'
	output_val_images_dir = output_images_dir + '/val'
	output_test_images_dir = output_images_dir + '/test'

	output_train_labels_dir = output_labels_dir + '/train'
	output_val_labels_dir = output_labels_dir + '/val'
	output_test_labels_dir = output_labels_dir + '/test'

	class_mapping = {}

	all_images = glob.glob(input_images_dir + '/*')
	all_graphs = glob.glob(input_lg_dir + '/*')

	random.seed(352)

	random.shuffle(all_images)

	train_size = int(len(all_images) * 0.7)
	val_size = int(len(all_images) * 0.15)
	test_size = int(len(all_images) * 0.15)

	# sample_images = [input_images_dir + '/00007.png']

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
			if symbol_class not in class_mapping:
				class_mapping[symbol_class] = len(class_mapping)

			bbox_yolo = convert_to_yolo_format(bbox[j][1:5], image_width, image_height)
			class_index = class_mapping.get(symbol_class, -1)

			if class_index != -1:
				# print((class_index,) + bbox_yolo)
				yolo_boxes.append((class_index,) + bbox_yolo)

		if i < train_size:
			img_dest_folder = output_train_images_dir
			lbl_dest_folder = output_train_labels_dir
		elif i < train_size + val_size:
			img_dest_folder = output_val_images_dir
			lbl_dest_folder = output_val_labels_dir
		else:
			img_dest_folder = output_test_images_dir
			lbl_dest_folder = output_test_labels_dir

		lbl_filename = lg_filename.replace('.lg', '.txt')

		shutil.copy(img_filepath , img_dest_folder + '/' + img_filename)
		with open(lbl_dest_folder + '/' + lbl_filename, 'w') as label:
			for yb in yolo_boxes:
				label.write(f"{yb[0]} {' '.join(str(coord) for coord in yb[1:5])}\n")

		if i % 100 == 0 :
			print(f"Completed {i} images..")

	with open(os.path.join(data_dir, 'class_mapping.txt'), 'w') as class_mapping_file:
		for symbol_class, index in class_mapping.items():
			class_mapping_file.write(f"{index}: {symbol_class}\n")


if __name__ == "__main__":
    main()





		
		







