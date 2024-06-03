import cv2
import os
import shutil
import glob

def move_video_missing_aligns_to_test(args):
	"""Moves all files which have missing align file to the target output_dir. usually the test directory

	Args:
			args (_type_): _description_
	"""
	assert args.output_dir != None, "Please supply a output dir"
	assert args.input_dir != None, "Please supply a input dir"
	video_ext = args.video_ext
	output_dir = args.output_dir

	for file_path in glob.glob(os.path.join(args.input_dir,  f'*.{video_ext}'), recursive=False):
		filename = os.path.basename(file_path)
		file_prefix = filename.split('.')[0]
		align_filename = file_prefix + ".align"
		align_file_path = os.path.join(os.path.dirname(file_path), align_filename)

		# Check if the .align file exists
		if not os.path.exists(align_file_path):
			# Move both files to the output directory
			shutil.move(file_path, os.path.join(output_dir, filename))
		else:
			print(f"Align file found for {filename}")

def split_val_from_train_using_crops(args):
	assert args.output_dir is not None, "Please supply an output dir"
	assert args.input_dir is not None, "Please supply an input dir"
	assert args.inputfile is not None, "Please supply an input filename using --inputfile"
	
	input_dir = args.input_dir
	output_dir = args.output_dir
	input_file = args.inputfile

	with open(input_file, 'r') as f:
		for line in f:
			filename = line.split('/')[0]  # Extract the filename before the first '/'
			video_file_path = os.path.join(input_dir, filename)
			align_file_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.align")
			
			# Check if the video file exists in the input directory
			if os.path.exists(video_file_path):
				shutil.move(video_file_path, os.path.join(output_dir, filename))
				
				# Check if the align file exists and move it
				if os.path.exists(align_file_path):
					shutil.move(align_file_path, os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.align"))
				else:
					print(f"Align file not found for {filename}")
			else:
				print(f"Video file not found: {filename}")

def move_files(args):
	"""Moves video+transcription files from one folder to another

	Args:
			args (_type_): _description_
	"""
	input_dir = args.input_dir if args.input_dir != None else "/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/datasets/grid/main/s1/"
	output_dir = args.output_dir if args.output_dir != None else "datasets/grid/pretrain/"

	# Put this in the args maybe, or just change inline
	files_to_move_file = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/unseen_train.txt'
	
	# Ensure the target directory exists
	os.makedirs(output_dir, exist_ok=True)

	# Read the list of files to move
	with open(files_to_move_file, 'r') as file:
		file_ids = file.readlines()
	
	# Loop through each file ID, remove the newline, and handle the path
	for file_id in file_ids:
		file_id = file_id.strip().split('/')[1]  # Remove any newline characters and spaces
		pattern = file_id + '*'  # Pattern to match all files starting with the file_id

		# Find all matching files and move them
		for filename in os.listdir(input_dir):
			if filename.startswith(file_id):
				src_path = os.path.join(input_dir, filename)
				dst_path = os.path.join(to_dir, filename)
				shutil.move(src_path, dst_path)
				print(f"Moved: {src_path} -> {dst_path}")