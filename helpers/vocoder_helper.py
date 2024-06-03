
from .util_helper import get_shape
import torch

import torch
import torchaudio
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle

def load_hifigan():
	vocoder = bundle.get_vocoder()
	print("Downloaded vocoder")

	specgram = torch.sin(0.5 * torch.arange(start=0, end=300)).expand(bundle._vocoder_params["in_channels"], 300)

	return vocoder



def hifigan_create_audio(hifigan, vocoder_train_setup, denoiser, mels):
	denoising_strength = 0.005
	audio = hifigan(mels).float()
	audio = denoiser(audio, denoising_strength)
	audio = audio * vocoder_train_setup['max_wav_value'] #audio.squeeze(1)
	return audio


def inverse_mel(hifigan, mel):
	print(f"Shape of mel before is {get_shape(mel)}")
	#  B,1,80,4S
	mels = mels.squeeze(1)
	print(f"Shape of mel after is {get_shape(mel)}")
	audios = hifigan_create_audio(hifigan,vocoder_train_setup, denoiser, mels)

	return audios


# #OLD HIFIGAN
# def load_hifigan():
# 	hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
# 	hifigan.cuda()
# 	denoiser.cuda()
# 	return hifigan, vocoder_train_setup, denoiser
# def inverse_mel(hifigan, vocoder_train_setup, denoiser, mels):
# 	print(f"Shape of mel before is {get_shape(mel)}")
# 	#  B,1,80,4S
# 	mels = mels.squeeze(1)
# 	print(f"Shape of mel after is {get_shape(mel)}")
# 	audios = hifigan_create_audio(hifigan,vocoder_train_setup, denoiser, mels)

# 	return audios