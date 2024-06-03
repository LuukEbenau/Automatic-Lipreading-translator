import cv2
import argparse
import os
import shutil
import glob

from preprocess import preprocess_video, extract_audio, process_unseen_file, move_files, move_video_missing_aligns_to_test,split_val_from_train_using_crops

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

	# , vocab_size=54
	spm.SentencePieceTrainer.train(input=f'{model_prefix}.temp.txt', vocab_size=55, model_prefix=model_prefix, user_defined_symbols=['sil'])

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

	tokenize_words(words_set, output_dir+"grid_lower2")

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
		
