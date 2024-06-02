import cv2
import argparse
from preprocess import preprocess_video
import os
import shutil
import glob
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--action", type=str, required=True, help="EXTRACT_AUDIO, TRAIN_LM, CREATE_UNSEEN_FILE, MOVE_FILES, CREATE_MOUTH_CROPS, MOVE_MISSING_ALIGNS, SPLIT_VAL_FROM_TRAIN")
	parser.add_argument('--outputfilename', type=str, help='Output file to write the coordinates')
	parser.add_argument('--input_dir', type=str, default=None, help='Output file to write the coordinates')
	parser.add_argument("--output_dir", default="./outputs/", type=str,help= "output")
	parser.add_argument("--video_ext", type=str, default="mpg")
	parser.add_argument("--audio_ext", type=str, default="wav")
	parser.add_argument("--inputfile", type=str)

	# for preprocessing
	args = parser.parse_args()
	return args

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

def crop_mouth(args):
	assert args.outputfilename != None, "please supply --outputfilename <output.txt>"
	output_path = args.output_dir

	preprocess_video(args.input_dir, args.video_ext, output_path, args.outputfilename)

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


def extract_audio(args):
	"""This function is not used right now, since we got the audio files predelivered. but in case we want to change samlingrate, etc. this might be useful!
	"""
	input_dir = args.input_dir if args.input_dir != None else '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/pretrain/'
	
	output_dir = args.output_dir if args.output_dir != None else '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID_audio/pretrain/'
	from_ext = args.video_ext
	to_ext = args.audio_ext 

	# Ensure the target directory exists
	os.makedirs(output_dir, exist_ok=True)
	import ffmpeg
	# Loop through all files in the from_dir
	for filename in os.listdir(input_dir):
		if filename.endswith(from_ext):
			full_file_path = os.path.join(input_dir, filename)
			output_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.' + to_ext)
			
			try:
				(
						ffmpeg
						.input(full_file_path)
						.output(output_file_path, acodec='pcm_s16le', ac=1, ar='16000')  # Set audio codec, channel, and sample rate
						.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
				)
				print(f"Extracted audio to {output_file_path}")
			except Exception as e:
				print(f"Failed to extract audio from {filename}. Error: {e}")



def extract_words_from_file(file_path):
	words = []
	with open(file_path, 'r') as file:
		for line in file:
			parts = line.split()
			if len(parts) == 3:
				word = parts[2]
				# if word != "sil":
				words.append(word)

	return words

def extract_words_from_folder(folder_path):
	import glob
	all_words = []
	for file_path in glob.glob(os.path.join(folder_path, '**', '*.align'), recursive=True):
		words = extract_words_from_file(file_path)
		all_words.extend(words)
	return all_words

def write_words_to_file(words, output_file_path):
	with open(output_file_path, 'w') as file:
		for word in words:
			file.write(word + '\n')

def tokenize_words(unique_words, model_prefix):
	import sentencepiece as spm
	with open(f'{model_prefix}.temp.txt', 'w') as f:
		f.write('\n'.join(unique_words))

	spm.SentencePieceTrainer.train(input=f'{model_prefix}.temp.txt', model_prefix=model_prefix, vocab_size=54, user_defined_symbols=['sil'])

	sp = spm.SentencePieceProcessor()
	sp.load(f'{model_prefix}.model')

	tokenized_words = [sp.encode_as_pieces(word) for word in unique_words]

	return tokenized_words

def train_language_model_GRID(args):
	output_dir = args.output_dir if args.output_dir != None else "D://Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/liptospeech_extern/data/GRID/"
	input_dir = args.input_dir if args.input_dir != None else "D://Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/datasets/alignments/alignments/"

	words = extract_words_from_folder(input_dir)
	print(f"Number of words: {len(words)}")
	words_set = set(words)
	print(f"Number of unique words {len(words_set)}")

	tokenize_words(words_set, output_dir+"grid_lower")



if __name__ == "__main__":
	args = parse_args()
	if args.action == "EXTRACT_AUDIO":
		extract_audio(args)
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
		
