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
#NOTE: Found some sources which also said converting it to decibels instead of amplitude works as decibels is logarithmic, and then this would work:
# power_to_db = librosa.power_to_db(spectrogram, ref=np.max) which I found from https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056
def mel_to_log(mel):
	log_spec = torch.clamp(mel, min=1e-10).log10()
	log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
	log_spec = (log_spec + 4.0) / 4.0
	return log_spec

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

	# reduce the audio channels dimension since we run it mono
	mel = torch.squeeze(mel, dim=1) 
	gen_mel = torch.squeeze(gen_mel, dim=1)
	# Add padding which is required by whisper
	mel = pad_mel_spectrogram(mel, target_length=3000)
	gen_mel = pad_mel_spectrogram(gen_mel, target_length=3000) 

	# Transforming to log mel, since this is what whisper is expecting
	mel = mel_to_log(mel)
	gen_mel = mel_to_log(gen_mel)

	batch_size = len(mel)
	# why [1,1], and not [batch_Size,1]?
	decoder_input_ids = torch.tensor([[1, 1]]) * whisper_model.config.decoder_start_token_id
	decoder_input_ids = decoder_input_ids.cuda()

	real_output_state = whisper_model(mel, decoder_input_ids=decoder_input_ids).last_hidden_state
	gen_output_state = whisper_model(gen_mel, decoder_input_ids=decoder_input_ids).last_hidden_state

	gen_ctc_loss = F.mse_loss(real_output_state, gen_output_state)
	return gen_ctc_loss