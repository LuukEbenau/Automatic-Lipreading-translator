import cv2
import argparse
import os
import shutil
import glob

def process_unseen_file(input_dir, output_dir, outputfilename, dataset_type, video_ext):
	result_file = os.path.join(output_dir, outputfilename) # "unseen_train.txt"

	with open(result_file, 'w') as f:
		for file_path in glob.glob(os.path.join(input_dir, f'*.{video_ext}')):
			filename = os.path.basename(file_path)
			file_prefix = os.path.splitext(filename)[0]
			f.write(f"{dataset_type}/{file_prefix}\n")