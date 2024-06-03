import cv2
import argparse
import os
import shutil
import glob

def process_unseen_file(args):
	assert args.input_dir is not None, "Please supply a --input_dir, to something like /liptospeech_extern/data/GRID/"
	assert args.output_dir is not None, "Please supply a --output_dir, to something like /liptospeech_extern/data/GRID/"
	input_dir = args.input_dir
	output_dir = args.output_dir

	result_file = os.path.join(output_dir, args.outputfilename) # "unseen_train.txt"

	with open(result_file, 'w') as f:
		video_ext = args.video_ext
		for file_path in glob.glob(os.path.join(input_dir, f'*.{video_ext}')):
			filename = os.path.basename(file_path)
			file_prefix = os.path.splitext(filename)[0]
			f.write(f"trainval/{file_prefix}\n")