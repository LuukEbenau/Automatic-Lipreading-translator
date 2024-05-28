import face_recognition
import cv2
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	# Include additional arguments if needed
	parser.add_argument('--output', type=str, default='output2.txt', help='Output file to write the coordinates')
	parser.add_argument('--data_dir', type=str, default='/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/datasets/grid/train/main/s1', help='Output file to write the coordinates')

	# for preprocessing
	parser.add_argument('--unseenfile', action='store_true')
	parser.add_argument('--extractaudio', action='store_true')
	parser.add_argument('--movefiles', action='store_true')
	args = parser.parse_args()
	return args

def main(args):
		# Define the video file path and the output path for bounding boxes
		output_path = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/outputs/'
		# video_file_path = 'swwc7a.mpg' 
		# video_path = args.data_dir + '/' + video_file_path
		# identifier = video_path.split('/')[-1].split('.')[0]  # This will be the video identifier without file extension

		from preprocess import preprocess_video
		preprocess_video(args.data_dir, 'mpg', output_path, args.output)

def process_unseen_file():
	unseen_train_file = "/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/unseen_train.txt"
	# unseen_test_file = "/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/unseen_test.txt"
	# unseen_val_file = "/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/unseen_val.txt"

	preprocess_pretrain_file = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/GRID_crop/preprocess_pretrain.txt'

	with open(unseen_train_file, 'w') as wf:
		with open(preprocess_pretrain_file, 'r') as file:
			for line in file:
				# Process each line here
				fn = line.strip().split('/')[0].split('.')[0]
				identifier = 'pretrain/'+fn
				wf.write(identifier+'\n')
				# print('file name part is',fn)

import os
import shutil

def move_files():
	from_dir = "/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/datasets/grid/main/s1/"
	to_dir = "datasets/grid/pretrain/"
	files_to_move_file = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/unseen_train.txt'
	
	# Ensure the target directory exists
	os.makedirs(to_dir, exist_ok=True)

	# Read the list of files to move
	with open(files_to_move_file, 'r') as file:
		file_ids = file.readlines()
	
	# Loop through each file ID, remove the newline, and handle the path
	for file_id in file_ids:
		file_id = file_id.strip().split('/')[1]  # Remove any newline characters and spaces
		pattern = file_id + '*'  # Pattern to match all files starting with the file_id

		# Find all matching files and move them
		for filename in os.listdir(from_dir):
			if filename.startswith(file_id):
				src_path = os.path.join(from_dir, filename)
				dst_path = os.path.join(to_dir, filename)
				shutil.move(src_path, dst_path)
				print(f"Moved: {src_path} -> {dst_path}")

import ffmpeg
def extract_audio():
	from_dir = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/pretrain/'
	to_dir = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID_audio/pretrain/'
	from_ext = 'mpg'
	to_ext = 'wav'

	# Ensure the target directory exists
	os.makedirs(to_dir, exist_ok=True)

	# Loop through all files in the from_dir
	for filename in os.listdir(from_dir):
		if filename.endswith(from_ext):
			full_file_path = os.path.join(from_dir, filename)
			output_file_path = os.path.join(to_dir, os.path.splitext(filename)[0] + '.' + to_ext)
			
			try:
				(
						ffmpeg
						.input(full_file_path)
						.output(output_file_path, acodec='pcm_s16le', ac=1, ar='25000')  # Set audio codec, channel, and sample rate
						.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
				)
				print(f"Extracted audio to {output_file_path}")
			except Exception as e:
				print(f"Failed to extract audio from {filename}. Error: {e}")

if __name__ == "__main__":
	args = parse_args()
	if args.unseenfile:
		process_unseen_file()
	elif args.extractaudio:
		extract_audio()
	elif args.movefiles:
		move_files()
	else:
		main(args)
