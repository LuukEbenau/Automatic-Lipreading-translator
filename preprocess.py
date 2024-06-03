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
	parser.add_argument("--inputfile", type=str)
	parser.add_argument("--samplerate", type=int, default=16000)

	# for preprocessing
	args = parser.parse_args()
	return args

def crop_mouth(args):
	assert args.outputfilename != None, "please supply --outputfilename <output.txt>"
	output_path = args.output_dir

	preprocess_video(args.input_dir, args.video_ext, output_path, args.outputfilename)



if __name__ == "__main__":
	args = parse_args()
	if args.action == "EXTRACT_AUDIO":
		assert args.input_dir is not None, 'Please supply input_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID/pretrain/'
		assert args.output_dir is not None, 'Please supply output_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID_audio/pretrain/'
		extract_audio(args.input_dir, args.output_dir, args.video_ext, args.audio_ext, args.samplerate)
	elif args.action == "CREATE_UNSEEN_FILE":
		process_unseen_file(args)
	elif args.action == "MOVE_FILES":
		move_files(args)
	elif args.action == "CREATE_MOUTH_CROPS":
		crop_mouth(args)
	elif args.action == "TRAIN_LM":
		train_language_model_GRID(args)
	elif args.action == "MOVE_MISSING_ALIGNS":
		move_video_missing_aligns_to_test(args)
	elif args.action == "SPLIT_VAL_FROM_TRAIN":
		split_val_from_train_using_crops(args)
	else:
		print(f"invalid action selected: {args.action}")
		
