import pkg_resources
import importlib
import torch
from .eval_helper import wer
def load_decoder(char_list):
	"""Load decoder based on installed software

	Args:
			char_list (string[]): tokens for the decoder, its a array of ngrams

	Raises:
			Exception: _description_

	Returns:
			decoder: decoder instance
	"""
	decodertype = get_decoder_type()
	if decodertype == 'torchaudio':
		from torchaudio.models.decoder import ctc_decoder
		# https://pytorch.org/audio/2.3.0/generated/torchaudio.models.decoder.ctc_decoder.html#torchaudio.models.decoder.ctc_decoder
		print(f"SIL token is {char_list[3]}")
		return ctc_decoder(
			lexicon=None,  # or specify a lexicon if needed
			tokens=char_list,
			lm=None,  # Language model, if any
			nbest=1,  # Number of best hypotheses to return
			beam_size=30,  # Beam search size (Zeyer et al., 2017).
			beam_threshold=80,  # Beam threshold (Graves et al., 2006).
			log_add=True , # Use log-add operation in beam search (Williams et al., 2006).
			blank_token = char_list[0], # should be sil character
			sil_token = char_list[3]
		)
		# https://arxiv.org/abs/1412.5567
	elif decodertype == "pyctcdecode":
		from pyctcdecode import build_ctcdecoder
		return build_ctcdecoder(
			labels=char_list,
			kenlm_model_path=None,
			alpha=0,
			beta=0,
			
			# beam_width=30,
			# num_cpus=4,
		)
	elif decodertype == "ctcdecode":
		from ctcdecode import CTCBeamDecoder
		return CTCBeamDecoder(
			char_list,
			model_path=None,
			alpha=0,
			beta=0,
			cutoff_top_n=40,
			cutoff_prob=1.0,
			beam_width=30,
			num_processes=4,
			blank_id=0,
			log_probs_input=False,
		)
	else:
		raise Exception(f"Decoder {decodertype} does not exist")

def decode_with_decoder(decoder, softmax_result, beam_wer, train_data, vid, target, blank_id):
	decodertype = get_decoder_type()

	if decodertype == "ctcdecode":
		beam_results, beam_scores, timesteps, out_lens = decoder.decode(softmax_result)
		beam_text = [train_data.arr2txt(beam_results[_][0][:out_lens[_][0]]) for _ in range(vid.size(0))]
	elif decodertype == "torchaudio":
		decoded_output = decoder(softmax_result)
		beam_results = []
		out_lens = []

		for hypotheses in decoded_output:
			batch_beam_results = []
			batch_out_lens = []
			for hypothesis in hypotheses:
				tokens = hypothesis.tokens.tolist()
				batch_beam_results.append(tokens)
				batch_out_lens.append(len(tokens))
			beam_results.append(batch_beam_results)
			out_lens.append(batch_out_lens)

		max_len = max(max(len(seq) for seq in batch) for batch in beam_results)
		beam_results_padded = [pad_sequences(batch, max_len, 0) for batch in beam_results]

		beam_results = torch.tensor(beam_results_padded)
		out_lens = torch.tensor(out_lens)

		beam_text = [train_data.arr2txt(torch.tensor(beam_results[_][0][:out_lens[_][0]])) for _ in range(vid.size(0))]
	else:
		raise Exception(f"Decoder {decodertype} does not exist")
	truth_txt = [train_data.arr2txt(target[_]) for _ in range(vid.size(0))]
	from .eval_helper import wer
	beam_wer.extend(wer(beam_text, truth_txt))
	return beam_text, truth_txt, beam_wer

def check_package_installed(package_name):
	# Check using pkg_resources
	is_installed = False
	try:
		pkg_resources.get_distribution(package_name)
		# print(f"{package_name} is installed (pkg_resources).")
		is_installed = True
	except pkg_resources.DistributionNotFound:
		pass
		# print(f"{package_name} is not installed (pkg_resources).")

	# Check using importlib
	try:
		importlib.import_module(package_name)
		# print(f"{package_name} is installed (importlib).")
		is_installed = True
	except ImportError:
		pass
		# print(f"{package_name} is not installed (importlib).")
	return is_installed

def get_decoder_type():
	if check_package_installed("torchaudio"):
		return "torchaudio"
	if check_package_installed("ctcdecode"):
		return "ctcdecode"
	elif check_package_installed("pyctcdecode"):
		return "pyctcdecode"
	else:
		print("WARNING: decoder detection failed, defaulting to pyctcdecode")
		return "pyctcdecode"

def pad_sequences(sequences, maxlen, padding_value):
	padded_sequences = []
	for seq in sequences:
		seq = seq + [padding_value] * (maxlen - len(seq))
		padded_sequences.append(seq)

	return padded_sequences

