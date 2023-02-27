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
from pathlib import Path
from tqdm import tqdm
import json

from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(eval_pred):
	"""Computes accuracy on a batch of predictions"""
	predictions = np.argmax(eval_pred.predictions, axis=1)   
	metric = load_metric("accuracy")                         
	return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def compute_metrics_macro_f1(eval_pred):
	"""Computes accuracy on a batch of predictions"""
	predictions = np.argmax(eval_pred.predictions, axis=1)
	metric = load_metric("f1")   
	
	# return metric.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')
	return metric.compute(predictions=predictions, references=eval_pred.label_ids, average='weighted')

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

		# breakpoint()

		# import IPython.display as ipd
		# ipd.Audio(data=np.asarray(audio_wav), autoplay=True, rate=16000)
		input_values = inputs.input_values.squeeze(0)
		return {'input_values'   : input_values, 
				'input_length'   : len(input_values),
				'attention_mask' : inputs.attention_mask.squeeze(0),
				'text': text,
				'labels': label}

	def __len__(self):
		return len(self.audios)

class LibriPhoneDataset(Dataset):
	def __init__(self, split, tokenizer, bucket_size, path, word2phonemes_path, ascending=False, **kwargs):
		# Setup
		self.path = path
		self.bucket_size = bucket_size

		with open(word2phonemes_path, "r") as f:
			word2phonemes = json.load(f)

		# List all wave files
		file_list = []
		for s in split:
			split_list = list(Path(join(path, s)).rglob("*.flac"))
			assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
			file_list += split_list
		
		text = []
		for f in tqdm(file_list, desc='word -> phonemes'):
			text.append(self.read_text(str(f), word2phonemes, tokenizer))

		self.file_list, self.text = zip(*[(f_name, txt)
										  for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

		# self.file_list = self.file_list[:100]
		# self.text = self.text[:100]
	
	def __getitem__(self, index):
		if self.bucket_size > 1:
			index = min(len(self.file_list)-self.bucket_size, index)
			return [(f_path, txt) for f_path, txt in
					zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
		else:
			return self.file_list[index], self.text[index]

	def __len__(self):
		return len(self.file_list)

	def read_text(self, file, word2phonemes, tokenizer):
		'''Get transcription of target wave file, 
		it's somewhat redundant for accessing each txt multiplt times,
		but it works fine with multi-thread'''
		src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
		idx = file.split('/')[-1].split('.')[0]

		with open(src_file, 'r') as fp:
			for line in fp:
				if idx == line.split(' ')[0]:
					transcription = line[:-1].split(' ', 1)[1]
					phonemes = []
					for word in transcription.split():
						phonemes += word2phonemes[word]
					return tokenizer.encode(' '.join(phonemes))



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
			# padding=self.padding,
			padding='max_length',
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
			truncation=True
		)

		assert batch["input_values"].size(1) == 200000

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
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length

def get_sv_vctk_data(data_dir, processor, mode):
	#from collections import Counter
	data_list = utils.get_vctk_files_no_overlap(data_dir, mode)[:512]
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	data_wavs, data_labels, sample_rate, all_labels = utils.read_sv_vstk_wav(data_list)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, all_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, data_list ,max_length


def get_emo_cls_iemocap_data(data_dir, processor, mode, wav_file_names, emotions):
	from collections import Counter
	data_list = utils.get_iemocap_files(data_dir, mode)
	if mode == "train":
		import random
		random.seed(100)
		random.shuffle(data_list)
	emotion_labels = ['sad', 'xxx', 'ang', 'fru', 'fea', 'exc', 'hap', 'sur', 'oth', 'neu', 'dis']

	data_wavs, data_labels, sample_rate = utils.read_emo_iemocap_wav(data_list, wav_file_names, emotions, emotion_labels)
	data_set = SpeechDataset(data_wavs, data_labels, processor, sample_rate, emotion_labels)
	max_length = max([len(d_wav) for d_wav in data_wavs])
	return data_set, max_length
	

def get_emo_meld_data(data_dir, processor, mode): #/data/path/MELD.Raw
	data_list = utils.get_meld_files(data_dir, mode)
	labels_dict = utils.get_emo_meld_label(data_dir, mode)
	emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']

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

def get_asr_esd_vocab_dict(data_dir):
	all_texts = utils.read_text(data_dir)
	def extract_all_chars(texts):
		all_text = " ".join(texts)
		vocab = list(set(all_text))
		return {"vocab":vocab}

	vocabs = extract_all_chars(all_texts)

	vocab_list = vocabs["vocab"]
	vocab_dict = {v: k for k, v in enumerate(vocab_list)}

	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	vocab_dict["[UNK]"] = len(vocab_dict)
	vocab_dict["[PAD]"] = len(vocab_dict)

	with open('vocab_esd.json', 'w') as vocab_file:
		json.dump(vocab_dict, vocab_file)

	

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
	# dataset, max_length = get_asr_data("/data/path/ESD/en/", processor, "test")
	# dataset = get_emo_cls_data("/data/path/ESD/en/", processor, "test")	

	# get_sp_vctk_data("/data/path/VCTK_Wav/wav48/", processor, "train")
	# get_emo_cls_iemocap_data("/data/path/IEMOCAP/IEMOCAP_full_release/", processor, "train")  #['train', 'evaluation', 'test']:
	# get_emo_meld_data("/data/path/MELD.Raw/", processor, "train")
	# get_emo_meld_data("/data/path/MELD.Raw/", processor, "evaluation")
	# get_emo_meld_data("/data/path/MELD.Raw/", processor, "test")

	# get_asr_meld_data("/data/path/MELD.Raw/", processor, "train")
	# get_asr_meld_data("/data/path/MELD.Raw/", processor, "evaluation")
	# get_asr_meld_data("/data/path/MELD.Raw/", processor, "test")
	# traindata , _= get_asr_fleurs_data('/data/path/fleurs', processor, "train")
	# validdata , _= get_asr_fleurs_data('/data/path/fleurs', processor, "evaluation")
	# testdata , _= get_asr_fleurs_data('/data/path/fleurs', processor, "test")

	# print("len of train:", len(traindata))
	# print("len of valid:", len(validdata))
	# print("len of test:", len(testdata))

	get_asr_voxpopuli_data('/data/path/voxpopuli', processor, "train")
	get_asr_voxpopuli_data('/data/path/voxpopuli', processor, "evaluation")
	get_asr_voxpopuli_data('/data/path/voxpopuli', processor, "test")


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
#   vctk_dataset = load_dataset("vctk", cache_dir='/data/path/VCTK')
# 	dir = "/data/path/VCTK/downloads/extracted/data/wav48_silence_trimmed"
# 	wavs = glob(os.path.join(dir, '*/*.flac'))
# 	pool = mp.Pool(processes = 64)
# 	with tqdm(total = len(wavs)) as pbar:
# 		for _ in tqdm(pool.imap_unordered(wav2pt, wavs)):
# 			pbar.update()
