import glob
import sentencepiece as spm

def train_language_model_GRID(input_dir, output_dir, outputfilename):
	words = extract_words_from_folder(input_dir)
	print(f"Number of words: {len(words)}")
	words_set = set(words)
	print(f"Number of unique words {len(words_set)}")

	tokenize_words(words_set, output_dir+outputfilename)

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
	all_words = []
	for file_path in glob.glob(os.path.join(folder_path, '**', '*.align'), recursive=True):
		words = extract_words_from_file(file_path)
		all_words.extend(words)
	return all_words

def tokenize_words(unique_words, model_prefix):
	
	with open(f'{model_prefix}.temp.txt', 'w') as f:
		f.write('\n'.join(unique_words))

	# , vocab_size=54
	spm.SentencePieceTrainer.train(input=f'{model_prefix}.temp.txt', vocab_size=55, model_prefix=model_prefix, user_defined_symbols=['sil'])

	sp = spm.SentencePieceProcessor()
	sp.load(f'{model_prefix}.model')

	tokenized_words = [sp.encode_as_pieces(word) for word in unique_words]

	return tokenized_words