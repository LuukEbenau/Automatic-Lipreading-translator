import cv2
import argparse
import os
import shutil
import glob

from preprocess import preprocess_video, extract_audio, process_unseen_file, move_files, move_video_missing_aligns_to_test,split_val_from_train_using_crops, train_language_model_GRID


'''
Create unseen file using:
python preprocess.py --action CREATE_UNSEEN_FILE --input_dir data/GRID_audio/pretrain/ --output_dir data/GRID/ --outputfilename unseen_train.txt

Create crops using:
python preprocess.py --action CREATE_MOUTH_CROPS --input_dir data/GRID/pretrain/ --output_dir data/GRID/GRID_crop/ --outputfilename preprocess_pretrain.txt
'''

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--action", type=str, required=True, help="EXTRACT_AUDIO, TRAIN_LM, CREATE_UNSEEN_FILE, MOVE_FILES, CREATE_MOUTH_CROPS, MOVE_MISSING_ALIGNS, SPLIT_VAL_FROM_TRAIN")
	parser.add_argument('--outputfilename', type=str, help='Output file to write the coordinates')
	parser.add_argument('--input_dir', type=str, default=None, help='Output file to write the coordinates')
	parser.add_argument("--output_dir", default="./outputs/", type=str,help= "output")
	parser.add_argument("--video_ext", type=str, default="mpg")
	parser.add_argument("--audio_ext", type=str, default="wav")
	parser.add_argument("--dataset_type", type=str, help="preprocess, trainval, test")
	parser.add_argument("--inputfile", type=str, help="If applicable, give the filename containing the input data")
	parser.add_argument("--samplerate", type=int, default=16000)

	# for preprocessing
	args = parser.parse_args()
	return args

def preprocess_all(args):
	dataset_name = args.dataset_name
	video_dir =  f'data/{dataset_name}/'
	audio_dir = f'data/{dataset_name}_audio/'

	# Do all actions neccesary. first, we assume that the input_dir is organised as following:
	# GRID -> [pretrain,trainval,test] -> [s1,s2,s3, or any other subfolder] -> [video (ext. args.video_ext), align(.align)]
	# GRID_audio ->  [pretrain,trainval,test] -> [s1,s2,s3, or any other subfolder] -> [.wav]
	# 1. Then, we use the alignments to construct the vocabulary.
	# 2. Then, we create the unseen files for each of the dataset types pretrain, trainval and test
	# 3. Then, we start with performing the face crops for each of the datasets
	
	
	train_language_model_GRID(input_dir = f"{video_dir}", output_dir = f"{video_dir}", outputfilename=f"{dataset_name.lower()}_lower")
	for dataset_type in ["pretrain", "trainval", "test"]:
		#TODO: WIP
		pass



if __name__ == "__main__":
	args = parse_args()
	if args.action == "EXTRACT_AUDIO":
		assert args.input_dir is not None, 'Please supply input_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID/pretrain/'
		assert args.output_dir is not None, 'Please supply output_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID_audio/pretrain/'
		extract_audio(args.input_dir, args.output_dir, args.video_ext, args.audio_ext, args.samplerate)
	
	elif args.action == "CREATE_UNSEEN_FILE":
		assert args.dataset_type != None, "Please supply a --dataset_type to pretrain, test or trainval"
		assert args.input_dir is not None, "Please supply a --input_dir, to something like /liptospeech_extern/data/GRID/"
		assert args.output_dir is not None, "Please supply a --output_dir, to something like /liptospeech_extern/data/GRID/"
		assert args.outputfilename is not None, "Please supply a outputfilename"
		process_unseen_file(args.input_dir, args.output_dir, args.outputfilename, args.dataset_type, args.video_ext)
	
	elif args.action == "MOVE_FILES":
		move_files(args)
	
	elif args.action == "CREATE_MOUTH_CROPS":
		assert args.outputfilename != None, "please supply --outputfilename <output.txt>"
		output_path = args.output_dir
		preprocess_video(args.input_dir, args.video_ext, output_path, args.outputfilename)

	elif args.action == "TRAIN_LM":
		assert args.input_dir is not None, "please supply --input_dir"
		assert args.output_dir is not None, "please supply --output_dir"
		train_language_model_GRID(args.input_dir, args.output_dir)

	elif args.action == "MOVE_MISSING_ALIGNS":
		move_video_missing_aligns_to_test(args)

	elif args.action == "SPLIT_VAL_FROM_TRAIN":
		split_val_from_train_using_crops(args)
	
	else:
		print(f"invalid action selected: {args.action}")
		
