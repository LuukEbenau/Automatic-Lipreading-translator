from torch.nn.functional import cross_entropy
import torch
import librosa
import torch

from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
import numpy as np
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor, WhisperModel
from torch.nn.functional import mse_loss

from torch.nn import functional as F

from .util_helper import get_shape
############################################## GENERAL ###############################################

# def mel_to_audio(mel_spectrogram, sr=16000):
# 	#	TODO: this function is from librosa. Is this maybe faster than the only that they used?
# 	# Convert Mel spectrogram to raw audio
# 	mel_spectrogram = mel_spectrogram.squeeze().numpy()
# 	audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
# 	return audio

def get_asr_model(asr_checkpoint_type, asr_checkpoint, num_characters):
	if asr_checkpoint is not None:
		if asr_checkpoint_type == "WHISPER":
			model_name = asr_checkpoint # "openai/whisper-base"
			asr_model = WhisperModel.from_pretrained(model_name).cuda()
			# asr_model = WhisperForConditionalGeneration.from_pretrained(model_name).cuda()
			# asr_processor = WhisperProcessor.from_pretrained(model_name)
		elif asr_checkpoint_type == "LRS2":
			asr_model = ASR_model(num_layers=6, num_attention_heads=4, num_class=num_characters)
		else:
			print("WARNING: no ASR selected")
			asr_model = None
	else:
		print("WARNING: no ASR selected")
		asr_model = None

	return asr_model

############################################### WHISPER ###############################################

def pad_mel_spectrogram(mel, target_length=3000):
	if mel.shape[-1] < target_length:
		pad_amount = target_length - mel.shape[-1]
		mel = torch.nn.functional.pad(mel, (0, pad_amount), 'constant', 0)
	return mel

def calculate_whisper_content_loss(mel, gen_mel, whisper_model, vocab_size, sampling_rate=16000):
	"""Calculate the content loss using whisper model
	
	Args:
			mel (_type_): mel spectrogram of the actual data
			gen_mel (_type_): generated mel spectrogram
			whisper_model (_type_): whisper model
	"""
	# print(f"shape of mel is {get_shape(mel)}")
	# print(f"shape of mel is {get_shape(gen_mel)}")

	mel = torch.squeeze(mel, dim=1)
	gen_mel = torch.squeeze(gen_mel, dim=1)
	mel = pad_mel_spectrogram(mel, target_length=3000)
	gen_mel = pad_mel_spectrogram(gen_mel, target_length=3000) #  pad_mel_spectrogram(gen_mel, target_length=3000)

	# mel = [pad_mel_spectrogram(mel[i], target_length=3000) for i in range(len(mel))] 
	# gen_mel = [pad_mel_spectrogram(gen_mel[i], target_length=3000) for i in range(len(mel))] #  pad_mel_spectrogram(gen_mel, target_length=3000)

	batch_size = len(mel)
	# print(f"Batch size is {batch_size}, and vocab size is {vocab_size}")
	# batch_size
	decoder_input_ids = torch.tensor([[1, 1]]) * whisper_model.config.decoder_start_token_id
	decoder_input_ids = decoder_input_ids.cuda()
	# decoder_input_ids = torch.tensor([[1, 1]]) * whisper_model.config.decoder_start_token_id
	# decoder_input_ids=decoder_input_ids
	real_output_state = whisper_model(mel, decoder_input_ids=decoder_input_ids).last_hidden_state
	gen_output_state = whisper_model(gen_mel, decoder_input_ids=decoder_input_ids).last_hidden_state

	gen_ctc_loss = F.mse_loss(real_output_state, gen_output_state)
	# print(f"ctc log is {gen_ctc_loss}")
	return gen_ctc_loss