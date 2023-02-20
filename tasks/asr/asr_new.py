
from datasets import load_dataset, load_metric#, Audio
# from datasets import ClassLabel
import random
import pandas as pd

import numpy as np
import torch

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import (set_seed, Wav2Vec2CTCTokenizer, 
		Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Config, 
		TrainingArguments, EarlyStoppingCallback, HfArgumentParser)
from transformers.integrations import TensorBoardCallback

# import sys 
# sys.path.append("..") 
from path import Path
import sys
folder = Path(__file__).abspath()
sys.path.append(folder.parent.parent.parent)

import os
from os.path import join

import utils
# from modules import CustomTrainer
from transformers import Trainer
# from transformers import Wav2Vec2ForCTC #as Wav2Vec2ForCTC_Transformer
from modeling_wav2vec2 import Wav2Vec2ForCTC
from data import get_asr_data, DataCollatorCTCWithPadding, get_asr_meld_data, get_asr_fleurs_data, get_asr_voxpopuli_data
from datasets import load_metric

from transformers.trainer_utils import EvalPrediction

from transformers.adapters.prefix_tuning import PrefixTuningPool
from transformers.adapters import PrefixTuningConfig, PfeifferInvConfig

import re
import json
import statistics

# def show_random_elements(dataset, processor, num_examples=10):
def show_random_elements(dataset, num_examples=10):
	assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
	picks = []
	for _ in range(num_examples):
		pick = random.randint(0, len(dataset)-1)
		while pick in picks:
			pick = random.randint(0, len(dataset)-1)
		picks.append(pick)

	keys = dataset[picks[0]].keys()
	for key in keys:
		print(key, end="		")
	print("\n")
	for idx in picks:
		for key in keys:
			print(dataset[idx][key], end=' ')
		print("\n")

	for idx in picks:
		print(processor.batch_decode(dataset[idx]["labels"]))

@dataclass
class DataTrainingArguments(TrainingArguments):
	dataset: Optional[str] = field(
		default="esd", metadata={"help": "dataset name"}
	)
	data_dir: Optional[str] = field(
		default="/data/yingting/ESD/en/", metadata={"help": "The dir of the dataset."}
	)
	feat_adapter_name: Optional[str] = field(
		default="conv_adapter", metadata={"help": "The type of adapter, should be chosen among in {conv_adapter }."}
	)
	trans_adapter_name: Optional[str] = field(
		default="bottleneck", metadata={"help": "The type of adapter, should be chosen among in {conv_adapter, bottleneck, adapterblock}."}
	)
	output_adapter: Optional[bool] = field(
		default=False, metadata={"help": "use adapter after FFN"}
	)
	mh_adapter: Optional[bool] = field(
		default=False, metadata={"help": "use adapter after multi-head attention"}
	)
	prefixtuning: Optional[bool] = field(
		default=False, metadata={"help": "use prefix-tuning in multi-head attention"}
	)
	prefix_tuning_my: Optional[bool] = field(
		default=False, metadata={"help": "use prefix-tuning in multi-head attention, implemented by us"}
	)
	prefix_seq_len: Optional[int] = field(
		default=30, metadata={"help": "prefix sequence length"}
	)
	prefix_projection: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Apply a two-layer MLP head over the prefix embeddings"
		}
	) 
	prefix_dropout_prob: Optional[bool] = field(
		default=0.1,
		metadata={
			"help": "The dropout probability used in the models"
		}
	)
	feat_enc_adapter: Optional[bool] = field(
		default=False, metadata={"help": "use conv_adapter in feature encoder and Adapterblock in  "}
	)
	lora_adapter: Optional[bool] = field(
		default=False, metadata={"help": "use lora_adapter in feature encoder"}
	)
	fine_tune: Optional[bool] = field(
		default=False, metadata={"help": "if fine-tune wav2vec2 or not"}
	)

def get_mean_length(dataset):
	all_lens = []
	for idx, _ in enumerate(dataset):
		all_lens.append(dataset[idx]["input_length"])
	return statistics.mean(all_lens)

def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]

	vocab_json = None

	# audio dataset
	if args.dataset.lower() == "esd":
		train_set, max_len_train = get_asr_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_asr_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_asr_data(args.data_dir, processor, "test")
	elif args.dataset.lower() == "meld":
		train_set, max_len_train = get_asr_meld_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_asr_meld_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_asr_meld_data(args.data_dir, processor, "test")
	elif args.dataset.lower() == "fleurs":

		fleurs = load_dataset("google/xtreme_s", "fleurs.en_us", cache_dir="/data/yingting/Dataset/fleurs")
		fleurs = fleurs.remove_columns(["num_samples", "raw_transcription", "gender", "lang_id","language", "lang_group_id"])

		chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
		def remove_special_characters(batch):
			batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower()
			return batch
		fleurs = fleurs.map(remove_special_characters)

		def extract_all_chars(batch):
			all_text = " ".join(batch["transcription"])
			vocab = list(set(all_text))
			return {"vocab": [vocab], "all_text": [all_text]}

		vocabs = fleurs.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=fleurs.column_names["train"])
		vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

		vocab_dict = {v: k for k, v in enumerate(vocab_list)}

		vocab_dict["|"] = vocab_dict[" "]
		del vocab_dict[" "]
		vocab_dict["[UNK]"] = len(vocab_dict)
		vocab_dict["[PAD]"] = len(vocab_dict)

		with open('vocab_fleurs.json', 'w') as vocab_file:
			json.dump(vocab_dict, vocab_file)

		vocab_json = 'vocab_fleurs.json'
		
		train_set = fleurs["train"]
		valid_set = fleurs["validation"]
		test_set = fleurs["test"]
	elif args.dataset.lower() == "voxpopuli":

		voxpopuli = load_dataset("facebook/voxpopuli", "en", cache_dir="/data/yingting/voxpopuli")
		voxpopuli = voxpopuli.remove_columns(["language", "raw_text", "gender", "speaker_id","is_gold_transcript", "accent"])

		chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
		def remove_special_characters(batch):
			batch["normalized_text"] = re.sub(chars_to_ignore_regex, '', batch["normalized_text"]).lower()
			return batch
		voxpopuli = voxpopuli.map(remove_special_characters)

		def extract_all_chars(batch):
			all_text = " ".join(batch["normalized_text"])
			vocab = list(set(all_text))
			return {"vocab": [vocab], "all_text": [all_text]}

		vocabs = voxpopuli.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=voxpopuli.column_names["train"])
		vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

		vocab_dict = {v: k for k, v in enumerate(vocab_list)}

		vocab_dict["|"] = vocab_dict[" "]
		del vocab_dict[" "]
		vocab_dict["[UNK]"] = len(vocab_dict)
		vocab_dict["[PAD]"] = len(vocab_dict)

		with open('vocab_voxpopuli.json', 'w') as vocab_file:
			json.dump(vocab_dict, vocab_file)

		vocab_json = 'vocab_voxpopuli.json'

		train_set = voxpopuli["train"]
		valid_set = voxpopuli["validation"]
		test_set = voxpopuli["test"]
	elif args.dataset.lower() == "librispeech":
		librispeech_train = load_dataset('librispeech_asr', 'clean', split='train.100', cache_dir='/data/yingting/hf_datasets')
		librispeech_dev = load_dataset('librispeech_asr', 'clean', split='validation', cache_dir='/data/yingting/hf_datasets')
		librispeech_test = load_dataset('librispeech_asr', 'clean', split='test', cache_dir='/data/yingting/hf_datasets')

		chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
		def remove_special_characters(batch):
			batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
			return batch
		librispeech_train = librispeech_train.map(remove_special_characters)
		librispeech_dev = librispeech_dev.map(remove_special_characters)
		librispeech_test = librispeech_test.map(remove_special_characters)

		def extract_all_chars(batch):
			all_text = " ".join(batch["text"])
			vocab = list(set(all_text))
			return {"vocab": [vocab], "all_text": [all_text]}

		vocabs_train = librispeech_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=librispeech_train.column_names)
		vocab_list = list(set(vocabs_train["vocab"][0]))

		vocab_dict = {v: k for k, v in enumerate(vocab_list)}

		vocab_dict["|"] = vocab_dict[" "]
		del vocab_dict[" "]
		vocab_dict["[UNK]"] = len(vocab_dict)
		vocab_dict["[PAD]"] = len(vocab_dict)

		with open('vocab_librispeech.json', 'w') as vocab_file:
			json.dump(vocab_dict, vocab_file)

		vocab_json = 'vocab_librispeech.json'

		train_set = librispeech_train
		valid_set = librispeech_dev
		test_set = librispeech_test
	else:
		raise NotImplementedError

	#processor
	tokenizer = Wav2Vec2CTCTokenizer(vocab_json, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
	
	# prepare dataset
	def prepare_dataset(batch):
		audio = batch["audio"]

		# batched output is "un-batched"
		batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
		# batch["input_values"] = processor(audio["array"], padding="True", max_length=160000, truncation=True, sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values[0]
		batch["input_length"] = len(batch["input_values"])
		
		with processor.as_target_processor():
			if args.dataset.lower() == "fleurs":
				batch["labels"] = processor(batch["transcription"]).input_ids
			elif args.dataset.lower() == "voxpopuli":
				batch["labels"] = processor(batch["normalized_text"]).input_ids
			elif args.dataset.lower() == "librispeech":
				batch["labels"] = processor(batch["text"]).input_ids
		return batch

	train_set = train_set.map(prepare_dataset, remove_columns=train_set.column_names)
	valid_set = valid_set.map(prepare_dataset, remove_columns=valid_set.column_names)
	test_set = test_set.map(prepare_dataset, remove_columns=test_set.column_names)

	# print("train mean len:", get_mean_length(train_set))
	# print("valid mean len:", get_mean_length(valid_set))
	# print("test  mean len:", get_mean_length(test_set))

	print("-----------------------------before filter-----------------------")
	print("----->>>> train")
	print(train_set)
	print("----->>>> valid")
	print(valid_set)
	print("----->>>> test")
	print(test_set)

	max_input_length_in_sec = 20.0
	train_set = train_set.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
	valid_set = valid_set.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
	test_set = test_set.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

	print("-----------------------------after filter-----------------------")
	print("----->>>> train")
	print(train_set)
	print("----->>>> valid")
	print(valid_set)
	print("----->>>> test")
	print(test_set)
	
	

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", vocab_size=len(processor.tokenizer))
	# config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m", vocab_size=len(processor.tokenizer))
	config._name_or_path = ""

	config.adapter_name = args.trans_adapter_name
	config.output_adapter = args.output_adapter
	config.mh_adapter = args.mh_adapter
	config.prefixtuning = args.prefixtuning
	config.prefix_tuning_my = args.prefix_tuning_my
	config.feat_enc_adapter = args.feat_enc_adapter
	config.lora_adapter = args.lora_adapter
	config.prefix_seq_len = args.prefix_seq_len
	config.prefix_projection = args.prefix_projection
	config.prefix_dropout_prob = args.prefix_dropout_prob
	config.ctc_loss_reduction = "mean"
	config.pad_token_id = processor.tokenizer.pad_token_id


	# load pretrained model
	model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config, ignore_mismatched_sizes=True)
	# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m", config=config, ignore_mismatched_sizes=True)
	model.freeze_feature_encoder()
	
	# print(model)

	print("\n #Train: {}, #Valid: {}, #Test: {} ".format(len(train_set), len(valid_set), len(test_set)))
	# print(" #Train Max len: {}, #Valid Max len: {}, #Test Max len: {} \n".format(max_len_train, max_len_valid, max_len_test))

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	# for name, param in model.named_parameters():
	# 	if param.requires_grad:
	# 		print(name, param.requires_grad, param.size())

	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, max_length=200000)
	wer_metric = load_metric("wer")

	def compute_metrics(pred):

		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)

		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

		pred_str = processor.batch_decode(pred_ids)
		# we do not want to group tokens when computing the metrics
		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

		wer = wer_metric.compute(predictions=pred_str, references=label_str)

		return {"wer": wer}

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=args,
		compute_metrics=compute_metrics,
		train_dataset=train_set,
		eval_dataset=valid_set,
		tokenizer=processor.feature_extractor,
		# callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
		callbacks = [TensorBoardCallback]
	)


	save_dir = join(args.output_dir, "best_model")
	# save_dir = join(args.output_dir, "checkpoint-1000")
	if args.do_train:   # train and test
		trainer.train(resume_from_checkpoint=None)	
		trainer.save_model(save_dir)

		test_metrics = trainer.predict(test_set).metrics
		print(test_metrics)

	if args.do_predict: # only for test
		from torch.utils.data import DataLoader
		device = trainer.model.device
		trainer.model = trainer.model.from_pretrained(save_dir).to(device)

		print(trainer.predict)

		test_metrics = trainer.predict(test_set).metrics
		print(test_metrics)
	

if __name__ == "__main__":
	main()