import cv2
import argparse
import os
import shutil
import glob

def process_unseen_file(input_dir, output_dir, outputfilename, dataset_type, video_ext):
	result_file = os.path.join(output_dir, outputfilename) # "unseen_train.txt"

	with open(result_file, 'w') as f:
		for file_path in glob.glob(os.path.join(input_dir, f'**/*.{video_ext}')):
			# Extract the relative path
			relative_path = os.path.relpath(file_path, input_dir)
			# Split into subdirectory and filename
			subdir, filename = os.path.split(relative_path)
			file_prefix = os.path.splitext(filename)[0]
			f.write(f"{dataset_type}/{subdir}/{file_prefix}\n")
			# filename = os.path.basename(file_path)
			# file_prefix = os.path.splitext(filename)[0]
			# f.write(f"{dataset_type}/{file_prefix}\n")
