import argparse
import os
import sys
import csv
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.csv_generator import _open_for_csv, _read_annotations, _open_for_csv, _read_classes
import keras
import tensorflow as tf

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)
def parse_args(args):
	""" Parse the arguments.
	"""
	parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
	subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
	subparsers.required = True

	coco_parser = subparsers.add_parser('coco')
	coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

	pascal_parser = subparsers.add_parser('pascal')
	pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

	csv_parser = subparsers.add_parser('csv')
	csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
	csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

	parser.add_argument('model',              help='Path to RetinaNet model.')
	parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
	parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
	parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
	parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
	parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
	parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
	parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
	parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
	parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
	parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')


	return parser.parse_args(args)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def main(args=None):
	# parse arguments
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# make sure keras is the minimum required version
	check_keras_version()

	# optionally choose specific GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	keras.backend.tensorflow_backend.set_session(get_session())

	# make save path if it doesn't exist
	if args.save_path is not None and not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	# optionally load config parameters
	if args.config:
		args.config = read_config_file(args.config)

	# create the generator
	#generator = create_generator(args)

	# optionally load anchor parameters
	anchor_params = None
	if args.config and 'anchor_parameters' in args.config:
		anchor_params = parse_anchor_parameters(args.config)

	# load the model
	print('Loading model, this may take a second...')
	model = models.load_model(args.model, backbone_name=args.backbone)

	# optionally convert the model
	if args.convert_model:
		model = models.convert_model(model, anchor_params=anchor_params)

	# print model summary
	print(model.summary())

	print("annotations",args.annotations)
	print("classes",args.classes)
	print("min",args.image_min_side)
	print("max",args.image_max_side)
	print("configs",args.config)
	iou_threshold=args.iou_threshold
	score_threshold=args.score_threshold
	max_detections=args.max_detections


	image_names = []
	image_data  = {}
	

	# Take base_dir from annotations file if not explicitly specified.
   

	# parse the provided class file
	try:
		with _open_for_csv(args.classes) as file:
			classes = _read_classes(csv.reader(file, delimiter=','))
	except ValueError as e:
		raise_from(ValueError('invalid CSV class file: {}: {}'.format(args.classes, e)), None)

	labels = {}
	for key, value in classes.items():
		labels[value] = key

	# csv with img_path, x1, y1, x2, y2, class_name
	try:
		with _open_for_csv(args.annotations) as file:
			file_annotations = _read_annotations(csv.reader(file, delimiter=','), classes)
	except ValueError as e:
		raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(args.annotations, e)), None)
	image_names = list(file_annotations.keys())
	
	num_classes = len(labels)

	all_detections = [[None for i in range(num_classes) if i in labels] for j in range(len(image_names))]
	for image_index in range(len(image_names)):
		""" Load annotations for an image_index.
		"""
		path        = file_annotations[image_names[image_index]]
		annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

		for idx, annot in enumerate(file_annotations[image_names[image_index]]):
			for key, value in classes.items():
				if annot['class'] == key:
					break;


			annotations['labels'] = np.concatenate((annotations['labels'], [value]))
			annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
				float(annot['x1']),
				float(annot['y1']),
				float(annot['x2']),
				float(annot['y2']),
			]]))

	f = []
	for label in  range(num_classes):
		for cls in classes:
				if classes[cls] == label:
					print('class',cls)
					break
		f.append(open("results/comp4_det_"+args.annotations.split("/")[-1].split(".")[0]+"_"+cls+".txt",'w'))


	for i in range(0, len(image_names)):
		print('image num',i)
		file_name = image_names[i]


		# load image
		image = read_image_bgr(file_name)
		image = preprocess_image(image)
		image, scale = resize_image(image)
		boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
		boxes /= scale

		# select indices which have a score above the threshold
		indices = np.where(scores[0, :] > score_threshold)[0]

		# select those scores
		scores = scores[0][indices]

		# find the order with which to sort the scores
		scores_sort = np.argsort(-scores)[:max_detections]

		# select detections
		image_boxes      = boxes[0, indices[scores_sort], :]
		image_scores     = scores[scores_sort]
		image_labels     = labels[0, indices[scores_sort]]
		image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
		# copy detections to all_detections
		for label in  range(num_classes):
			if not label in labels:
				continue
			dets = image_detections[image_detections[:, -1] == label, :-1]
			for cls in classes:
				if classes[cls] == label:
					print('class',cls)
					break
			for scr in dets:
				f[label].write(file_name.split("/")[-1].split(".")[0]+" "+str(scr[4])+" "+str(scr[0])+" "+str(scr[1])+" "+str(scr[2])+" "+str(scr[3])+"\n")
				
	for label in  range(num_classes):
		f[label].close()
	


if __name__ == '__main__':
	main()