from transformers import Wav2Vec2Processor
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils

import torch
import os
import re
from os.path import join
from glob import glob
from torch import nn
import random
import torch.nn.functional as F

# import librosa as rosa
import librosa
# from omegaconf import OmegaConf as OC
from tqdm import tqdm
import multiprocessing as mp

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import ClassLabel
import pandas as pd
# from IPython.display import display, HTML

import soundfile as sf

def compute_metrics(eval_pred):
	"""Computes accuracy on a batch of predictions"""
	predictions = np.argmax(eval_pred.predictions, axis=1)
	metric = load_metric("accuracy")
	return metric.compute(predictions=predictions, references=eval_pred.label_ids)

class SpeechDataset(Dataset):
	def __init__(self, audios, labels, processor: Wav2Vec2Processor, sample_rate, all_labels):
		self.audios = audios
		self.labels = labels
		self.processor = processor
		self.sample_rate = sample_rate
		label2id, id2label = dict(), dict()
		for i, label in enumerate(all_labels):
			label2id[label] = str(i)
			id2label[str(i)] = label

		self.num_labels = len(all_labels)
		self.label2id = label2id
		self.id2label = id2label
	
	def __getitem__(self, index):
		audio_wav = self.audios[index]
		
		inputs = self.processor(audio_wav, padding="max_length", max_length=40000, truncation=True, sampling_rate=self.sample_rate, return_tensors="pt")

		label = self.labels[index]

		return {'input_values':inputs.input_values.squeeze(0), 
				'attention_mask':inputs.attention_mask.squeeze(0),
				'labels': label}
	
	def __len__(self):
		return len(self.audios)

class AsrDataset(Dataset):
	def __init__(self, audios, texts, processor: Wav2Vec2Processor, sample_rate):
		self.audios = audios
		self.texts = texts
		self.processor = processor
		self.sample_rate = sample_rate
		self.chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

	def remove_special_characters(self, text):
		text = re.sub(self.chars_to_ignore_regex, '', text).lower()
		return text	

	def __getitem__(self, index):
		audio_wav = self.audios[index]
		text = self.texts[index]
		text = self.remove_special_characters(text)
		inputs = self.processor(audio_wav, padding="max_length", max_length=40000, truncation=True, sampling_rate=self.sample_rate, return_tensors="pt")

		with self.processor.as_target_processor():
			label = self.processor(text).input_ids

		# import IPython.display as ipd
		# ipd.Audio(data=np.asarray(audio_wav), autoplay=True, rate=16000)
		return {'input_values'   : inputs.input_values.squeeze(0), 
				'attention_mask' : inputs.attention_mask.squeeze(0),
				'text': text,
				'labels': label}

	def __len__(self):
		return len(self.audios)



class DataCollatorCTCWithPadding:
	"""
	Data collator that will dynamically pad the inputs received.
	Args:
		processor (:class:`~transformers.Wav2Vec2Processor`)
			The processor used for proccessing the data.
		padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
			Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
			among:
			* :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
			  sequence if provided).
			* :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
			  maximum acceptable input length for the model if that argument is not provided.
			* :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
			  different lengths).
		max_length (:obj:`int`, `optional`):
			Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
		max_length_labels (:obj:`int`, `optional`):
			Maximum length of the ``labels`` returned list and optionally padding length (see above).
		pad_to_multiple_of (:obj:`int`, `optional`):
			If set will pad the sequence to a multiple of the provided value.
			This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
			7.5 (Volta).
	"""
	def __init__(self, 
				processor: Wav2Vec2Processor, 
				padding: Union[bool, str] = True, 
				max_length: Optional[int] = None, 
				max_length_labels: Optional[int] = None,
				pad_to_multiple_of: Optional[int] = None,
				pad_to_multiple_of_labels: Optional[int] = None):

		self.processor = processor
		self.padding = padding
		self.max_length = max_length
		self.max_length_labels = max_length_labels
		self.pad_to_multiple_of = pad_to_multiple_of
		self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# split inputs and labels since they have to be of different lenghts and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				max_length=self.max_length_labels,
				pad_to_multiple_of=self.pad_to_multiple_of_labels,
				return_tensors="pt",
			)

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch



def get_data(data_dir, processor, mode):
	data_list = utils.get_file_list(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_labels, sample_rate, all_labels = utils.read_wav(data_list)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_sp_cls_data(data_dir, processor, mode):
	data_list = utils.get_file_list(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_labels, sample_rate, all_labels = utils.read_sp_cls_wav(data_list)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_sp_vctk_data(data_dir, processor, mode):
	from collections import Counter

	data_list = utils.get_vctk_files(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_labels, sample_rate, all_labels = utils.read_sp_vstk_wav(data_list)
	tmp = np.array(data_labels)
	tmp = list(np.squeeze(tmp))
	print(Counter(tmp))
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length


def get_emo_cls_iemocap_data(data_dir, processor, mode, wav_file_names, emotions):
	from collections import Counter
	data_list = utils.get_iemocap_files(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	emotion_labels = ['sad', 'xxx', 'ang', 'fru', 'fea', 'exc', 'hap', 'sur', 'oth', 'neu', 'dis']

	data_wavs, data_labels, sample_rate = utils.read_emo_iemocap_wav(data_list, wav_file_names, emotions, emotion_labels)
	tmp = np.array(data_labels)
	tmp = list(np.squeeze(tmp))
	print(Counter(tmp))
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, emotion_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length
	

def get_emo_meld_data(data_dir, processor, mode): #/data/yingting/MELD.Raw
	data_list = utils.get_meld_files(data_dir, mode)
	labels_dict = utils.get_emo_meld_label(data_dir, mode)
	emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
	
	print("len of data_list:", len(data_list))

	data_wavs, data_labels, sample_rate = utils.read_emo_meld_wav(data_list, labels_dict, emotion_labels)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, emotion_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length
	

def get_ks_cls_data(data_dir, processor, mode):
	data_list = utils.get_ks_file_list(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_labels, sample_rate, all_labels = utils.read_ks_cls_wav(data_list)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_asr_data(data_dir, processor, mode):
	data_list = utils.get_file_list(data_dir, mode)
	all_texts = utils.read_text(data_dir)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_texts, sample_rate = utils.read_asr_wav(data_list, all_texts)
	data_set = AsrDataset(data_wavs, data_texts, processor, sample_rate)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_asr_meld_data(data_dir, processor, mode):
	data_list = utils.get_meld_files(data_dir, mode)
	utterances_dict = utils.read_meld_text(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_texts, sample_rate = utils.read_asr_meld_wav(data_list, utterances_dict)
	data_set = AsrDataset(data_wavs, data_texts, processor, sample_rate)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_asr_fleurs_data(data_dir, processor, mode):
	from datasets import load_dataset
	fleurs = load_dataset("google/xtreme_s", "fleurs.en_us", cache_dir=data_dir)

	data_wavs = []
	data_texts = []
	# print(fleurs.keys())   #['train', 'validation', 'test']

	if mode == "train":
		fleurs_asr = fleurs["train"]
	elif mode == "evaluation":
		fleurs_asr = fleurs["validation"]
	elif mode == "test":
		fleurs_asr = fleurs["test"]

	for i,sample in enumerate(fleurs_asr):
		data_wavs.append(fleurs_asr[i]["audio"]['array'])
		data_texts.append(fleurs_asr[i]["transcription"])

	sample_rate = fleurs_asr[0]["audio"]['sampling_rate']

	data_set = AsrDataset(data_wavs, data_texts, processor, sample_rate)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

	# audio_input = fleurs["train"][0]["audio"]  # first decoded audio sample
	# transcription = fleurs["train"][0]["transcription"]
	# # wav_input, sample_rate = sf.read(audio_input['path'])
	# print(audio_input.keys())    #['path', 'array', 'sampling_rate']
	# print(audio_input['path'])
	# print(audio_input['array'])
	# print(wav_input)
	# print(audio_input['sampling_rate'])
	# print(transcription)

def get_asr_voxpopuli_data(data_dir, processor, mode):
	from datasets import load_dataset
	voxpopuli_croatian = load_dataset("facebook/voxpopuli", "en", cache_dir=data_dir)

	data_wavs = []
	data_texts = []
	# empty_count = 0
	# empty_norm_count = 0
	if mode == "train":
		voxpopuli_asr = voxpopuli_croatian["train"]
	elif mode == "evaluation":
		voxpopuli_asr = voxpopuli_croatian["validation"]
	elif mode == "test":
		voxpopuli_asr = voxpopuli_croatian["test"]


	for i,sample in enumerate(voxpopuli_asr):
		# if voxpopuli_asr[i]["raw_text"] == "":
		# 	empty_count += 1
		# if voxpopuli_asr[i]["normalized_text"] == "":
		# 	empty_norm_count += 1
		if voxpopuli_asr[i]["normalized_text"] != "":
			data_wavs.append(voxpopuli_asr[i]["audio"]['array'])
			data_texts.append(voxpopuli_asr[i]["normalized_text"])

	sample_rate = voxpopuli_asr[0]["audio"]['sampling_rate']

	data_set = AsrDataset(data_wavs, data_texts, processor, sample_rate)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

if __name__=="__main__":
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	# dataset, max_length = get_asr_data("/data/yingting/ESD/en/", processor, "test")
	# dataset = get_emo_cls_data("/data/yingting/ESD/en/", processor, "test")	

	# get_sp_vctk_data("/data/yingting/VCTK_Wav/wav48/", processor, "train")
	# get_emo_cls_iemocap_data("/data/yingting/IEMOCAP/IEMOCAP_full_release/", processor, "train")  #['train', 'evaluation', 'test']:
	# get_emo_meld_data("/data/yingting/MELD.Raw/", processor, "train")
	# get_emo_meld_data("/data/yingting/MELD.Raw/", processor, "evaluation")
	# get_emo_meld_data("/data/yingting/MELD.Raw/", processor, "test")

	# get_asr_meld_data("/data/yingting/MELD.Raw/", processor, "train")
	# get_asr_meld_data("/data/yingting/MELD.Raw/", processor, "evaluation")
	# get_asr_meld_data("/data/yingting/MELD.Raw/", processor, "test")
	# traindata , _= get_asr_fleurs_data('/data/yingting/fleurs', processor, "train")
	# validdata , _= get_asr_fleurs_data('/data/yingting/fleurs', processor, "evaluation")
	# testdata , _= get_asr_fleurs_data('/data/yingting/fleurs', processor, "test")

	# print("len of train:", len(traindata))
	# print("len of valid:", len(validdata))
	# print("len of test:", len(testdata))

	get_asr_voxpopuli_data('/data/yingting/voxpopuli', processor, "train")
	get_asr_voxpopuli_data('/data/yingting/voxpopuli', processor, "evaluation")
	get_asr_voxpopuli_data('/data/yingting/voxpopuli', processor, "test")

	


# if __name__=="__main__":
	
	# processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	# esd_dataset, _ = get_data("/data/yingting/ESD/en/", processor, "test")
	# print("esd_dataset: \n", esd_dataset[0])
	# print(type(esd_dataset[0]["labels"][0]))

	# vctk_dataset = VCTKMultiSpkDataset("/data/yingting/VCTK/downloads/extracted/data/wav48_silence_trimmed", processor, 16000, cv=0)
	# print("vctk_dataset: \n", vctk_dataset[0])
	# print(type(vctk_dataset[0]["labels"][0]))
	


##### process VCTK data from .flac to .pt


# def wav2pt(wav):
# 	y,_ = rosa.load(wav, sr = 48000, mono = True)
# 	y,_ = rosa.effects.trim(y, 15)
# 	pt_name = os.path.splitext(wav)[0]+'.pt'
# 	pt = torch.tensor(y)
# 	torch.save(pt ,pt_name)
# 	del y, pt 
# 	return

# if __name__=='__main__':
#   vctk_dataset = load_dataset("vctk", cache_dir='/data/yingting/VCTK')
# 	dir = "/data/yingting/VCTK/downloads/extracted/data/wav48_silence_trimmed"
# 	wavs = glob(os.path.join(dir, '*/*.flac'))
# 	pool = mp.Pool(processes = 64)
# 	with tqdm(total = len(wavs)) as pbar:
# 		for _ in tqdm(pool.imap_unordered(wav2pt, wavs)):
# 			pbar.update()

# class LowPass(nn.Module):
# 	def __init__(self,
# 				 nfft=1024,
# 				 hop=256,
# 				 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
# 						1 / 1)):
# 		super().__init__()
# 		self.nfft = nfft
# 		self.hop = hop
# 		self.register_buffer('window', torch.hann_window(nfft), False)
# 		f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
# 		for i, r in enumerate(ratio):
# 			f[i, int((nfft//2+1) * r):] = 0.
# 		self.register_buffer('filters', f, False)

# 	#x: [B,T], r: [B], int
# 	@torch.no_grad()
# 	def forward(self, x, r):
# 		if x.dim()==1:
# 			x = x.unsqueeze(0)
# 		T = x.shape[1]
# 		x = F.pad(x, (0, self.nfft), 'constant', 0)
# 		stft = torch.stft(x,
# 						  self.nfft,
# 						  self.hop,
# 						  window=self.window,
# 						  )#return_complex=False)  #[B, F, TT,2]
# 		stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
# 		x = torch.istft(stft,
# 						self.nfft,
# 						self.hop,
# 						window=self.window,
# 						)#return_complex=False)
# 		x = x[:, :T].detach()
# 		return x

# class VCTKMultiSpkDataset(Dataset):
# 	def __init__(self, data_dir, processor: Wav2Vec2Processor, sample_rate, cv=0):  #cv 0: train, 1: val, 2: test
# 		def _get_datalist(folder, file_format, spk_list, cv):
# 			_dl = []
# 			len_spk_list = len(spk_list)
# 			s=0
# 			print(f'full speakers {len_spk_list}')
# 			for i, spk in enumerate(spk_list):
# 				if cv==0:
# 					if not(i<int(len_spk_list*self.cv_ratio[0])): continue
# 				elif cv==1:
# 					if not(int(len_spk_list*self.cv_ratio[0])<=i and
# 							i<=int(len_spk_list*(self.cv_ratio[0]+self.cv_ratio[1]) )):
# 						continue
# 				else:
# 					if not(int(len_spk_list*self.cv_ratio[0])<=i and
# 							i<=int(len_spk_list*(self.cv_ratio[0]+self.cv_ratio[1]) )):
# 						continue
# 				_full_spk_dl = sorted(glob(os.path.join(spk, file_format)))
# 				_len = len(_full_spk_dl)
# 				if (_len == 0): continue
# 				s+=1	
# 				_dl.extend(_full_spk_dl)
			
# 			print(cv, s)
# 			return _dl

# 		def _get_spk(folder):
# 			return sorted(glob(os.path.join(folder, '*')))#[1:])
		
# 		self.cv = cv
# 		self.cv_ratio = eval(str((100./108., 8./108., 0.00)))
# 		self.directory = data_dir #"/data/yingting/VCTK/downloads/extracted/data/wav48_silence_trimmed"
# 		self.dataformat = '*mic1.pt'
# 		self.data_list = _get_datalist(self.directory, self.dataformat,
# 									   _get_spk(self.directory), self.cv)
# 		self.processor = processor
# 		self.sample_rate = sample_rate

# 		# self.filter_ratio = [1./2]
# 		# self.lowpass = LowPass(1024,
# 		# 					   256,
# 		# 					   ratio=self.filter_ratio)
# 		# self.upsample = torch.nn.Upsample(scale_factor=2,
# 		# 								  mode ='linear',
# 		# 								  align_corners = False)
# 		assert len(self.data_list) != 0, "no data found"
# 		# self.get_all_labels()
# 		speaker_labels = set(['p279', 'p317', 'p266', 'p303', 'p339', 'p254', 'p230', 'p298', \
# 			'p333', 'p307', 'p274', 'p273', 'p340', 'p295', 'p236', 'p238', 'p241', 'p252', 'p248', \
# 			'p313', 'p297', 'p287', 'p253', 'p229', 'p233', 'p234', 'p268', 'p306', 'p261', 'p323', \
# 			'p232', 'p282', 'p286', 'p269', 'p329', 'p250', 'p284', 'p270', 'p237', 'p283', 'p300', \
# 			'p244', 'p292', 'p345', 'p256', 'p263', 'p304', 'p311', 'p285', 'p262', 'p316', 'p341', \
# 			'p258', 'p265', 'p305', 'p257', 'p249', 'p259', 'p260', 'p302', 'p245', 'p288', 'p227', \
# 			'p294', 'p334', 'p264', 'p271', 'p267', 'p330', 'p275', 'p240', 'p281', 'p299', 'p318', \
# 			'p243', 'p293', 'p277', 'p272', 'p308', 'p276', 'p310', 'p326', 'p239', 'p225', 'p226', \
# 			'p335', 'p347', 'p343', 'p278', 'p228', 'p231', 'p314', 'p247', 'p312', 'p255', 'p336', \
# 			'p246', 'p251', 'p301', 's5', 'p360', 'p362', 'p374', 'p351', 'p364', 'p361', 'p363', 'p376'])
		
# 		label2id, id2label = dict(), dict()
# 		for i, label in enumerate(speaker_labels):
# 			label2id[label] = str(i)
# 			id2label[str(i)] = label

# 		self.num_labels = len(speaker_labels)
# 		self.label2id = label2id
# 		self.id2label = id2label

# 	def __len__(self):
# 		return len(self.data_list)

# 	def get_all_labels(self):
# 		speakers = set ()
# 		for data_f in self.data_list:
# 			speaker = data_f.split("/")[-2]
# 			speakers.add(speaker)
# 		return list(speakers)

# 	def __getitem__(self, index):
# 		wav = torch.load(self.data_list[index])
# 		wav /= wav.abs().max()
# 		if wav.shape[0] < 32768:
# 			padl = 32768 - wav.shape[0]
# 			r = random.randint(0, padl) if self.cv<2 else padl//2
# 			wav = torch.nn.functional.pad(wav, (r, padl-r), 'constant', 0)
# 		else:
# 			start = random.randint(0, wav.shape[0] - 32768)
# 			wav = wav[start:start+32768] if self.cv<2 \
# 					else wav[:len(wav)-len(wav)%2]
# 		wav *= random.random()/2+0.5 if self.cv<2 else 1

# 		# wav_l = self.lowpass(wav, 0)
# 		# wav_l = wav_l[0,::2].view(1,1,-1)
# 		# #or
# 		# #wav_l = rosa.resample(wav, hparams.audio.sr, hparams.audio.sr//hparams.audio.ratio)
# 		# wav_l = self.upsample(wav_l).view(1,-1)
# 		# return wav, wav_l

# 		inputs = self.processor(wav, padding="max_length", max_length=40000, truncation=True, sampling_rate=self.sample_rate, return_tensors="pt")

# 		label = [int(self.label2id[self.data_list[index].split("/")[-2]])]

# 		return {'input_values':inputs.input_values.squeeze(0), 
# 				'attention_mask':inputs.attention_mask.squeeze(0),
# 				'labels': label}

# def get_sp_vctk_data_old(data_dir, processor, mode):
# 	data_list = utils.get_vctk_files(data_dir, mode)
# 	if mode == "train":
# 		import random
# 		random.seed(100)
# 		random.shuffle(data_list)
# 	speaker_labels = ['p279', 'p317', 'p266', 'p303', 'p339', 'p254', 'p230', 'p298', \
# 			'p333', 'p307', 'p274', 'p273', 'p340', 'p295', 'p236', 'p238', 'p241', 'p252', 'p248', \
# 			'p313', 'p297', 'p287', 'p253', 'p229', 'p233', 'p234', 'p268', 'p306', 'p261', 'p323', \
# 			'p232', 'p282', 'p286', 'p269', 'p329', 'p250', 'p284', 'p270', 'p237', 'p283', 'p300', \
# 			'p244', 'p292', 'p345', 'p256', 'p263', 'p304', 'p311', 'p285', 'p262', 'p316', 'p341', \
# 			'p258', 'p265', 'p305', 'p257', 'p249', 'p259', 'p260', 'p302', 'p245', 'p288', 'p227', \
# 			'p294', 'p334', 'p264', 'p271', 'p267', 'p330', 'p275', 'p240', 'p281', 'p299', 'p318', \
# 			'p243', 'p293', 'p277', 'p272', 'p308', 'p276', 'p310', 'p326', 'p239', 'p225', 'p226', \
# 			'p335', 'p347', 'p343', 'p278', 'p228', 'p231', 'p314', 'p247', 'p312', 'p255', 'p336', \
# 			'p246', 'p251', 'p301', 'p360', 'p362', 'p374', 'p351', 'p364', 'p361', 'p363', 'p376']
# 	data_labels = []
# 	for i in range(len(data_list)):
# 		label = [speaker_labels.index(str(data_list[i].split(',')[0].split('/')[-2]))]
# 		data_labels.append(label)

# 	data_set = SpeechClsDataset(data_list, data_labels, processor, speaker_labels)
# 	return data_set

# def get_emo_cls_iemocap_data_old(data_dir, processor, mode, wav_file_names, emotions, all_labels):
# 	data_list = utils.get_iemocap_files(data_dir, mode)
# 	if mode == "train":
# 		import random
# 		random.seed(100)
# 		random.shuffle(data_list)
# 	# print(data_list[0])
# 	# print("len of data_list:", len(data_list))
# 	# # wav_file_names, emotions = utils.get_iemocap_labels(data_dir)
# 	# # all_labels = list(set(emotions))
# 	# # print("all_labels:", all_labels)
# 	labels = []
# 	for data_file in data_list:
# 		wav_file_name = data_file.split("/")[-1][:-4]
# 		idx = wav_file_names.index(wav_file_name)
# 		labels.append(all_labels.index(str(emotions[idx])))
# 	data_set = SpeechClsDataset(data_list, labels, processor, all_labels)
# 	return data_set

# def get_emo_cls_data(data_dir, processor, mode):   ### read wav when need, no big diffference in running time
# 	data_list = utils.get_file_list(data_dir, mode)
# 	if mode == "train":
# 		import random
# 		random.seed(100)
# 		random.shuffle(data_list)
# 	emo_label = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']
# 	data_labels = []
# 	for i in range(len(data_list)):
# 		label = [emo_label.index(str(data_list[i].split(',')[0].split('/')[-3]))]
# 		data_labels.append(label)
# 	data_set = SpeechClsDataset(data_list, data_labels, processor, emo_label)
# 	return data_set

# class SpeechClsDataset(Dataset):
# 	def __init__(self, audios_files, labels, processor: Wav2Vec2Processor, cls_labels):
# 		self.audios_files = audios_files
# 		self.labels = labels
# 		self.processor = processor
		
# 		label2id, id2label = dict(), dict()
# 		for i, label in enumerate(cls_labels):
# 			label2id[label] = str(i)
# 			id2label[str(i)] = label

# 		self.num_labels = len(cls_labels)
# 		self.label2id = label2id
# 		self.id2label = id2label
	
# 	def __getitem__(self, index):
# 		# audio_wav = self.audios[index]
# 		audio_file = self.audios_files[index]
# 		# audio_wav, sample_rate = sf.read(audio_file) 
# 		audio_wav, sample_rate = librosa.load(audio_file, sr=16000)
		
# 		inputs = self.processor(audio_wav, padding=True, max_length=40000, truncation=True, sampling_rate=sample_rate, return_tensors="pt")

# 		label = self.labels[index]

# 		return {'input_values':inputs.input_values.squeeze(0), 
# 				'attention_mask':inputs.attention_mask.squeeze(0),
# 				'labels': label}
	
# 	def __len__(self):
# 		return len(self.audios_files)

# class VCTKSpeechDataset(Dataset):
# 	def __init__(self, audios, labels, processor: Wav2Vec2Processor, sample_rate, label_dict):
# 		self.audios = audios
# 		self.labels = labels
# 		self.processor = processor
# 		self.sample_rate = sample_rate
# 		# label2id, id2label = dict(), dict()
# 		# for i, label in enumerate(all_labels):
# 		# 	label2id[label] = str(i)
# 		# 	id2label[str(i)] = label


# 		self.num_labels = len(label_dict)
# 		self.label2id = label_dict
# 		self.id2label = {v:k for k,v in label_dict.items()}
	
# 	def __getitem__(self, index):
# 		audio_wav = self.audios[index]
		
# 		inputs = self.processor(audio_wav, padding="max_length", max_length=40000, truncation=True, sampling_rate=self.sample_rate, return_tensors="pt")

# 		label = self.labels[index]

# 		return {'input_values':inputs.input_values.squeeze(0), 
# 				'attention_mask':inputs.attention_mask.squeeze(0),
# 				'labels': label}
	
# 	def __len__(self):
# 		return len(self.audios)


