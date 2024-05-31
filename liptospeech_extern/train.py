import os
import time
import glob
import copy
import random
import argparse
from functools import partial

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad
from torch.nn import functional as F
from torch.nn import DataParallel as DP
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import librosa
from pesq import pesq
from pystoi import stoi
from matplotlib import pyplot as plt
import tqdm

from src.models.model import Visual_front, Conformer_encoder, CTC_classifier, Speaker_embed, Mel_classifier
from src.models.asr_model import ASR_model
#NOTE: These are functions i wrote myself, as convenience and to not clutter the train.py file
from src.helpers import decode_with_decoder, get_dataset, get_shape, load_decoder, wer, calculate_whisper_content_loss, get_asr_model, log_time

# TODO LIST:
# 1. Create vocabolary from dataset
# 2. fix CTCdecoder blank token

# diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)
# 3. replace mel_layer with flow matching 
# Compute loss of the decoder after mel layer
# calculate diffusion loss
# https://pytorch.org/audio/main/generated/torchaudio.pipelines.Wav2Vec2Bundle.html#torchaudio.pipelines.Wav2Vec2Bundle

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', default="Data_dir")
	parser.add_argument('--data_name', default="LRS2", help='LRS2, LRS3, GRID')
	parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/')
	parser.add_argument("--visual_front_checkpoint", type=str, default=None)
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--asr_checkpoint", type=str, default=None)

	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=150)
	parser.add_argument("--lr", type=float, default=0.0001)
	parser.add_argument("--weight_decay", type=float, default=0.00001)
	parser.add_argument("--workers", type=int, default=5)
	parser.add_argument("--seed", type=int, default=1)

	parser.add_argument("--eval_step", type=int, default=1000)

	parser.add_argument("--start_epoch", type=int, default=0)
	parser.add_argument("--augmentations", default=True)
	parser.add_argument("--mask_prob", type=float, default=0.4)

	parser.add_argument("--min_window_size", type=int, default=50)
	parser.add_argument("--max_window_size", type=int, default=50)
	parser.add_argument("--mode", type=str, default='train', help='train, test, val')
	parser.add_argument("--max_timesteps", type=int, default=250)

	parser.add_argument("--conf_layer", type=int, default=12)
	parser.add_argument("--num_head", type=int, default=8)

	parser.add_argument("--dataparallel", default=False, action='store_true')
	parser.add_argument("--output_content_loss", default=False, action='store_true')
	parser.add_argument("--output_content_on", type=float, default=0.7)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--asr_checkpoint_type", type=str, default="LRS2", help="LRS2, HUBERT, WHISPER")

	parser.add_argument("--samplerate", type=int, default=16000)
	args = parser.parse_args()
	return args

def train_net(args):
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)
	os.environ['OMP_NUM_THREADS'] = '6' #it was 2, can i make it bigger? 2
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	train_data = get_dataset(args)

	v_front = Visual_front(in_channels=1, conf_layer=args.conf_layer, num_head=args.num_head)

	# So modify mel classifier with the encoder from matchaTTS
	mel_layer = Mel_classifier()
	sp_layer = Speaker_embed()
	ctc_layer = CTC_classifier(train_data.num_characters)

	# Load asr model
	asr_model = get_asr_model(args.asr_checkpoint_type, args.asr_checkpoint,train_data.num_characters)

	if args.visual_front_checkpoint is not None:
		print(f"Loading checkpoint: {args.visual_front_checkpoint}")
		checkpoint = torch.load(args.visual_front_checkpoint, map_location=lambda storage, loc: storage.cuda())
		v_front.load_state_dict(checkpoint, strict=False)
		del checkpoint

	if args.checkpoint is not None:
		print(f"Loading checkpoint: {args.checkpoint}")
		checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
		v_front.load_state_dict(checkpoint['v_front_state_dict'])
		ctc_layer.load_state_dict(checkpoint['ctc_layer_state_dict'])
		mel_layer.load_state_dict(checkpoint['mel_layer_state_dict'])
		sp_layer.load_state_dict(checkpoint['sp_layer_state_dict'])
		del checkpoint

	if args.asr_checkpoint is not None and args.asr_checkpoint_type == "LRS2" or args.asr_checkpoint_type == "LRS3":
		print(f"Loading ASR checkpoint: {args.asr_checkpoint}")
		checkpoint = torch.load(args.asr_checkpoint, map_location=lambda storage, loc: storage.cuda())
		asr_model.load_state_dict(checkpoint['asr_model_state_dict'])
		del checkpoint

	v_front.cuda()
	mel_layer.cuda()
	sp_layer.cuda()
	ctc_layer.cuda()
	if args.asr_checkpoint is not None:
		asr_model.cuda()

	params = [{'params': v_front.parameters()},
			  {'params': mel_layer.parameters()},
			  {'params': sp_layer.parameters()},
			  {'params': ctc_layer.parameters()}]

	optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

	if args.dataparallel:
		v_front = DP(v_front)
		mel_layer = DP(mel_layer)
		sp_layer = DP(sp_layer)
		ctc_layer = DP(ctc_layer)
		if args.asr_checkpoint is not None:
			asr_model = DP(asr_model)

	# _ = validate(v_front, mel_layer, post, sp_layer, fast_validate=True)
	train(v_front, mel_layer, ctc_layer, sp_layer, asr_model, train_data, args.epochs, optimizer=optimizer, args=args, samplerate = args.samplerate)

def collate_data(data, batch):
    return data.collate_fn(batch)

def train(v_front, mel_layer, ctc_layer, sp_layer, asr_model, train_data, epochs, optimizer, args, samplerate = None):
	best_val_stoi = 0
	writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])

	v_front.train()
	mel_layer.train()
	sp_layer.train()
	ctc_layer.train()
	if args.asr_checkpoint is not None:
		asr_model.eval()

	collate_fn_partial = partial(collate_data, train_data)
	dataloader = DataLoader(
		train_data,
		shuffle=True,
		batch_size=args.batch_size,
		num_workers=args.workers,
		pin_memory=True,
		drop_last=True,
		collate_fn=collate_fn_partial,
	)

	decoder = load_decoder(train_data.char_list)

	stft = copy.deepcopy(train_data.stft).cuda()

	criterion = nn.L1Loss().cuda()
	CTC_criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum').cuda()
	samples = len(dataloader.dataset)
	batch_size = dataloader.batch_size
	step = 0

	for epoch in tqdm.tqdm(range(args.start_epoch, epochs)):
		recon_loss_list = []
		loss_list = []
		beam_wer = []
		print(f"Epoch [{epoch}/{epochs}]")
		prev_time = time.time()
		for i, batch in enumerate(dataloader):
			step += 1
			if i % 100 == 0:
				iter_time = (time.time() - prev_time) / 100
				prev_time = time.time()
				print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
					epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['lr']))
			mel, spec, vid, vid_len, wav_tr, mel_len, target, target_len, start_frame, window_size, f_name, sp_mel = batch
			max_window_size = window_size.max()

			##### For masked prediction
			#vid: B, C, T, H, W
			mel_masks = []
			for bb in range(vid.size(0)):
				if random.random() < args.mask_prob:
					mask_len = min(random.randint(2, 7), vid_len[bb] - start_frame[bb] - 1)   # span length
					mask_st_frame = random.randint(start_frame[bb], start_frame[bb] + window_size[bb] - mask_len)
					vid[bb, :, mask_st_frame:mask_st_frame + mask_len] = 0.
				mel_masks.append([1] * mel_len[bb] + [0] * (mel.size(3) - mel_len[bb]))

			mel_masks = torch.FloatTensor(mel_masks).view(vid.size(0), 1, 1, -1).cuda()    # B, 1, 1, S

			sp_feat = sp_layer(sp_mel.cuda())
			v_feat = v_front(vid.cuda(), vid_len.cuda())   # B,T,512
			ctc_pred = ctc_layer(v_feat)

			gen_v_feat = torch.zeros(v_feat.size(0), max_window_size, v_feat.size(2)).cuda()
			gen_v_len = []
			##### selecting window for generation ####
			for bb in range(vid.size(0)):
				temp_v_feat = v_feat[bb, start_frame[bb]:start_frame[bb] + window_size[bb], :]
				gen_v_feat[bb, :window_size[bb]] = temp_v_feat
				gen_v_len.append(temp_v_feat.size(0))
			gen_v_len = torch.tensor(gen_v_len).unsqueeze(1)

			gen_mel = mel_layer(gen_v_feat, sp_feat)  # B,1,80,4S
			gen_mel = gen_mel * mel_masks

			################################### GEN ########################################
			if args.asr_checkpoint is not None and args.output_content_loss:
				if args.asr_checkpoint_type == "LRS2" or args.asr_checkpoint_type == "LRS3":
					_, gen_ctc_feat = asr_model(gen_mel.cuda(), gen_v_len)
					_, real_ctc_feat = asr_model(mel, gen_v_len)
					gen_ctc_loss = F.mse_loss(gen_ctc_feat, real_ctc_feat)

				elif args.asr_checkpoint_type == "WHISPER":
					# with log_time("CALCULATING WHISPER LOSS"):
					# wav_pred = train_data.inverse_mel(gen_mel.cuda().detach(), mel_len, stft)  # 1, 80, T
					# wav_gt = train_data.inverse_mel(mel.cuda().detach(), mel_len, stft)
					gen_ctc_loss = calculate_whisper_content_loss(mel.cuda(), gen_mel.cuda(), asr_model, train_data.char_list)
					# print("Whisper content loss:", gen_ctc_loss)
				else:
					gen_ctc_loss = torch.zeros(1).cuda()
			else:
				gen_ctc_loss = torch.zeros(1).cuda()

			recon_loss = criterion(train_data.denormalize(gen_mel), train_data.denormalize(mel.cuda()))

			ctc_loss = CTC_criterion(ctc_pred.transpose(0, 1).log_softmax(2), target.cuda(), vid_len, target_len) / vid.size(0)

			gen_loss = 100.0 * recon_loss + ctc_loss + gen_ctc_loss

			loss_list.append(gen_loss.cpu().item())
			recon_loss_list.append(recon_loss.cpu().item())

			optimizer.zero_grad()
			gen_loss.backward()
			optimizer.step()

			################################### DECODE ########################################
			softmax_result = F.softmax(ctc_pred, 2).cpu()
			beam_text, truth_txt, beam_wer = decode_with_decoder(decoder, softmax_result, beam_wer, train_data, vid, target, train_data.char_list[0])

			################################### VISUALIZE & VALIDATE ########################################
			if i % 100 == 0:
				wav_pred = train_data.inverse_mel(gen_mel.detach()[0], mel_len[0:1], stft)  # 1, 80, T
				wav_gt = train_data.inverse_mel(mel.cuda().detach()[0], mel_len[0:1], stft)
			else:
				wav_pred = 0
				wav_gt = 0

			if writer is not None:
				writer.add_scalar('train/recon_loss', recon_loss.cpu().item(), step)
				writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]['lr'], step)
				writer.add_scalar('train/ctc_loss', ctc_loss.cpu(), step)
				writer.add_scalar('train/g_ctc_loss', gen_ctc_loss.cpu(), step)
				if i % 100 == 0:
					print(f'######## Step(Epoch): {step}({epoch}), Recon Loss: {recon_loss.cpu().item()}, Ctc loss: {ctc_loss.cpu()}, ASR loss {gen_ctc_loss.cpu()} #########')
					for (predict, truth) in list(zip(beam_text, truth_txt))[:3]:
						print(f'VP: {predict.upper()}')
						print(f'GT: {truth.upper()}\n') # Ground truth
					writer.add_scalar('train/wer', np.array(beam_wer).mean(), step)
					writer.add_image('train_mel/gen', train_data.plot_spectrogram_to_numpy(gen_mel.cpu().detach().numpy()[0]), step)
					writer.add_image('train_mel/gt', train_data.plot_spectrogram_to_numpy(mel.detach().numpy()[0]), step)
					writer.add_audio('train_aud/pred_mel', wav_pred[0], global_step=step, sample_rate=args.samplerate)
					writer.add_audio('train_aud/gt_mel', wav_gt[0], global_step=step, sample_rate=args.samplerate)
					writer.add_audio('train_aud/gt_wav', wav_tr[0].numpy(), global_step=step, sample_rate=args.samplerate)

			if step % args.eval_step == 0:
				logs = validate(v_front, mel_layer, sp_layer, args, epoch=epoch, writer=writer, fast_validate=True)

				print('VAL_stoi: ', logs[1])
				print('Saving checkpoint: %d' % epoch)
				if args.dataparallel:
					v_state_dict = v_front.module.state_dict()
					mel_layer_state_dict = mel_layer.module.state_dict()
					sp_layer_state_dict = sp_layer.module.state_dict()
					ctc_layer_state_dict = ctc_layer.module.state_dict()
				else:
					v_state_dict = v_front.state_dict()
					mel_layer_state_dict = mel_layer.state_dict()
					sp_layer_state_dict = sp_layer.state_dict()
					ctc_layer_state_dict = ctc_layer.state_dict()
				if not os.path.exists(args.checkpoint_dir):
					os.makedirs(args.checkpoint_dir)
				torch.save({'v_front_state_dict': v_state_dict, 'ctc_layer_state_dict': ctc_layer_state_dict,
							'mel_layer_state_dict': mel_layer_state_dict, 'sp_layer_state_dict': sp_layer_state_dict},
						   os.path.join(args.checkpoint_dir, 'Epoch_%04d_stoi_%.3f_estoi_%.3f_pesq_%.3f.ckpt' % (
						   epoch, logs[1], logs[2], logs[3])))

				if logs[1] > best_val_stoi:
					best_val_stoi = logs[1]
					bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
					for prev in bests:
						os.remove(prev)
					torch.save({'v_front_state_dict': v_state_dict, 'ctc_layer_state_dict': ctc_layer_state_dict,
								'mel_layer_state_dict': mel_layer_state_dict,
								'sp_layer_state_dict': sp_layer_state_dict},
							   os.path.join(args.checkpoint_dir, 'Best_%04d_stoi_%.3f_estoi_%.3f_pesq_%.3f.ckpt' % (
							   epoch, logs[1], logs[2], logs[3])))

		if np.mean(recon_loss_list) < args.output_content_on and epoch > 4:
			args.output_content_loss = True

	print('Finishing training')

def validate(v_front, mel_layer, sp_layer,args, fast_validate=True, epoch=0, writer=None):
	with torch.no_grad():
		v_front.eval()
		mel_layer.eval()
		sp_layer.eval()

		val_data = get_dataset(args, val=True)

		collate_fn_partial = partial(collate_data, val_data)
		dataloader = DataLoader(
			val_data,
			shuffle=True if fast_validate else False,
			batch_size=args.batch_size,
			num_workers=args.workers,
			drop_last=False,
			collate_fn=collate_fn_partial,
		)

		stft = copy.deepcopy(val_data.stft).cuda()
		criterion = nn.L1Loss().cuda()
		batch_size = dataloader.batch_size
		
		if fast_validate:
			samples = min(10 * batch_size, int(len(dataloader.dataset)))
			max_batches = 10
		else:
			samples = int(len(dataloader.dataset))
			max_batches = int(len(dataloader))

		val_loss = []
		stoi_list = []
		estoi_list = []
		pesq_list = []

		required_iter = (samples // batch_size)

		description = 'Validation on subset of the Val dataset' if fast_validate else 'Validation'
		print(description)
		for i, batch in enumerate(dataloader):
			if i % 10 == 0:
				if not fast_validate:
					print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
			mel, spec, vid, vid_len, wav_tr, mel_len, targets, target_len, _, _, f_name, _ = batch

			sp_feat = sp_layer(mel[:, :, :, :50].cuda())
			v_feat = v_front(vid.cuda(), vid_len.cuda())  # S,B,512

			g_mel = mel_layer(v_feat, sp_feat)

			loss = criterion(g_mel, mel.cuda()).cpu().item()
			val_loss.append(loss)

			wav_pred = val_data.inverse_mel(g_mel, mel_len, stft)
			wav_gt = val_data.inverse_mel(mel.cuda(), mel_len, stft)
			for _ in range(g_mel.size(0)):
				min_len = min(len(wav_pred[_]), len(wav_tr[_]))
				stoi_list.append(stoi(wav_tr[_][:min_len].numpy(), wav_pred[_][:min_len], args.samplerate, extended=False))
				estoi_list.append(stoi(wav_tr[_][:min_len].numpy(), wav_pred[_][:min_len], args.samplerate, extended=True))
				try:
					wav_tr_resampled = librosa.resample(wav_tr[_][:min_len].numpy(), args.samplerate, 8000)
					wav_pred_resampled = librosa.resample(wav_pred[_][:min_len], args.samplerate, 8000)			

					print(f"Shape of wav_tr_resampled: {get_shape(wav_tr_resampled)}, Shape of wav_pred_resampled: {get_shape(wav_pred_resampled)}")
					print(wav_tr_resampled[0])
					print(wav_pred_resampled[0])
					pesq_score = pesq(8000, wav_tr_resampled, wav_pred_resampled , 'nb')

					pesq_list.append(pesq_score)
				except:
					pass

			if i in [int(required_iter // 3), int(2 * (required_iter // 3)), int(3 * (required_iter // 3))]:
				if writer is not None:
					writer.add_image('val_mel_%d/gen' % i, val_data.plot_spectrogram_to_numpy(
						g_mel.cpu().detach().numpy()[0][:, :, :mel_len[0]]), epoch)
					writer.add_image('val_mel_%d/gt' % i,
									 val_data.plot_spectrogram_to_numpy(mel.detach().numpy()[0][:, :, :mel_len[0]]),
									 epoch)

					writer.add_audio('val_aud_%d/pred' % i, wav_pred[0], global_step=epoch, sample_rate=args.samplerate)
					writer.add_audio('val_aud_%d/mel' % i, wav_gt[0][:len(wav_pred[0])], global_step=epoch,
									 sample_rate=args.samplerate)
					writer.add_audio('val_aud_%d/gt' % i, wav_tr[0][:len(wav_pred[0])], global_step=epoch,
									 sample_rate=args.samplerate)
					fig = plt.figure()
					ax = fig.add_subplot(1, 1, 1)
					ax.set(xlim=[0, len(wav_pred[0])], ylim=[-1, 1])
					ax.plot(wav_pred[0])
					writer.add_figure('val_wav_%d/pred_mel' % i, fig, epoch)
					fig = plt.figure()
					ax = fig.add_subplot(1, 1, 1)
					ax.set(xlim=[0, len(wav_gt[0][:len(wav_pred[0])])], ylim=[-1, 1])
					ax.plot(wav_gt[0])
					writer.add_figure('val_wav_%d/mel' % i, fig, epoch)
					fig = plt.figure()
					ax = fig.add_subplot(1, 1, 1)
					ax.set(xlim=[0, len(wav_tr[0][:len(wav_pred[0])])], ylim=[-1, 1])
					ax.plot(wav_tr[0])
					writer.add_figure('val_wav_%d/gt' % i, fig, epoch)

			if i >= max_batches:
				break

		if writer is not None:
			writer.add_scalar('val/recon_loss', np.mean(np.array(val_loss)), epoch)
			writer.add_scalar('val/mel_stoi', np.mean(np.array(stoi_list)), epoch)
			writer.add_scalar('val/mel_estoi', np.mean(np.array(estoi_list)), epoch)
			writer.add_scalar('val/mel_pesq', np.mean(np.array(pesq_list)), epoch)

		v_front.train()
		sp_layer.train()
		mel_layer.train()

		print('val_stoi:', np.mean(np.array(stoi_list)))
		print('val_estoi:', np.mean(np.array(estoi_list)))
		print('val_pesq:', np.mean(np.array(pesq_list)))
		return np.mean(np.array(val_loss)), np.mean(np.array(stoi_list)), np.mean(np.array(estoi_list)), np.mean(np.array(pesq_list))





if __name__ == "__main__":
	args = parse_args()
	train_net(args)