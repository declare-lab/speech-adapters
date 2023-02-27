from datasets import load_dataset, load_metric

import re
import json
import torch

from path import Path
import sys
from os.path import join
folder = Path(__file__).abspath()
sys.path.append(folder.parent.parent.parent)

from data import LibriPhoneDataset, DataCollatorCTCWithPadding
from text import load_text_encoder
from transformers import Trainer

from transformers import (set_seed, Wav2Vec2CTCTokenizer, 
		Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Config, 
		TrainingArguments, EarlyStoppingCallback, HfArgumentParser)

from modeling_wav2vec2 import Wav2Vec2ForCTC

from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import datasets
from datasets import Dataset
from torch.distributed import is_initialized
from typing import Optional
from dataclasses import field, dataclass

from transformers.integrations import TensorBoardCallback

@dataclass
class DataTrainingArguments(TrainingArguments):
	dataset: Optional[str] = field(
		default="esd", metadata={"help": "dataset name"}
	)
	data_dir: Optional[str] = field(
		default="/data/path/ESD/en/", metadata={"help": "The dir of the dataset."}
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
	prefix_tuning: Optional[bool] = field(
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

def seed_worker(_):
	"""
	Helper function to set worker seed during Dataloader initialization.
	"""
	worker_seed = torch.initial_seed() % 2**32
	set_seed(worker_seed)

class CustomTrainer(Trainer):
	def get_train_dataloader(self) -> DataLoader:
		"""
		Returns the training [`~torch.utils.data.DataLoader`].
		Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
		training if necessary) otherwise.
		Subclass and override this method if you want to inject some custom behavior.
		"""
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")
		train_dataset = self.train_dataset

		train_sampler = DistributedSampler(dataset) if is_initialized() else None

		collate_fn = partial(self.collect_audio_batch, split="train")

		return DataLoader(
			train_dataset,
			batch_size=self._train_batch_size,
			sampler=train_sampler,
			# collate_fn=data_collator,
			collate_fn=collate_fn,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			worker_init_fn=seed_worker,
		)

	def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
		"""
		Returns the evaluation [`~torch.utils.data.DataLoader`].
		Subclass and override this method if you want to inject some custom behavior.
		Args:
			eval_dataset (`torch.utils.data.Dataset`, *optional*):
				If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
				by the `model.forward()` method are automatically removed. It must implement `__len__`.
		"""
		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		collate_fn = partial(self.collect_audio_batch, split="dev")

		return DataLoader(
			eval_dataset,
			# sampler=eval_sampler,
			shuffle=False,
			batch_size=self.args.eval_batch_size,
			collate_fn=collate_fn,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
		)

	def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
		"""
		Returns the test [`~torch.utils.data.DataLoader`].
		Subclass and override this method if you want to inject some custom behavior.
		Args:
			test_dataset (`torch.utils.data.Dataset`, *optional*):
				The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
				`model.forward()` method are automatically removed. It must implement `__len__`.
		"""
		data_collator = self.data_collator

		collate_fn = partial(self.collect_audio_batch, split="test")

		# We use the same batch_size as for eval.
		return DataLoader(
			test_dataset,
			batch_size=self.args.eval_batch_size,
			collate_fn=collate_fn,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
		)

	def collect_audio_batch(self, batch, split, half_batch_size_wav_len=300000):
		'''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
		e.g. [(file1,txt1),(file2,txt2),...]
		'''
		def audio_reader(filepath):
			wav, sample_rate = torchaudio.load(filepath)
			return wav.reshape(-1)

		# Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
		if type(batch[0]) is not tuple:
			batch = batch[0]

		# Make sure that batch size is reasonable
		first_len = audio_reader(str(batch[0][0])).size(0)
		if split == 'train':
			if first_len > half_batch_size_wav_len and len(batch) > 1:
				batch = batch[:len(batch)//2]

		# Read batch
		file, audio_feat, audio_len, text = [], [], [], []
		with torch.no_grad():
			for b in batch:
				file.append(str(b[0]).split('/')[-1].split('.')[0])
				feat = audio_reader(str(b[0])).numpy()
				audio_feat.append(feat)
				audio_len.append(len(feat))
				text.append(torch.LongTensor(b[1]).numpy())

		# Descending audio length within each batch
		audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
												for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])
		
		# return audio_feat, text, file

		labels = [torch.FloatTensor(label) for label in text]
		labels = pad_sequence(labels, padding_value=-100).transpose(0,1)

		wavs = [torch.FloatTensor(wav) for wav in audio_feat]
		wavs = pad_sequence(wavs).transpose(0,1)

		return {"input_values":wavs,
				"labels":labels}
	

# tokenizer
tokenizer = load_text_encoder(mode="word", vocab_file="phoneme.txt")

wer_metric = load_metric("wer")

def compute_metrics(pred):

	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)

	pred.label_ids[pred.label_ids == -100] = tokenizer._vocab2idx["<pad>"]

	pred.label_ids = pred.label_ids.astype(int)

	pred_str = [[tokenizer.decode(seq)] for seq in pred_ids]
	label_str = [[tokenizer.decode(seq.tolist())] for seq in pred.label_ids]

	per = wer_metric.compute(predictions=pred_str, references=label_str)

	return {"per": per}

def main():
	set_seed(1314)
	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]



	# audio dataset
	if args.dataset.lower() == "timit":
		timit = load_dataset("timit_asr", cache_dir="/data/path/Dataset/timit_asr/")
	elif args.dataset.lower() == "librispeech":
		train_path = "/data/path/hf_datasets/downloads/extracted/baf2e051c7d5c26b3b25db6157338d0eca8b961c9f49f25f65e10b0d583678e1/LibriSpeech"
		dev_path = "/data/path/hf_datasets/downloads/extracted/d89a8a1d668652cbb712b0970ff79b3e200655cf354aa6e8b87660ee441a7edf/LibriSpeech"
		test_path = "/data/path/hf_datasets/downloads/extracted/f6e39073841bee74aaa6f25d34420963669676bf57915cf6ad2403a7a833df68/LibriSpeech"
		word2phonemes_path = "/home/path/PromptSpeech/tasks/phoneme_recognition/word2phonemes.json"

		kwargs = {'num_workers': 24, 'train': ['train-clean-100'], 'dev': ['dev-clean'], 'test': ['test-clean']}

		dev_dataset = LibriPhoneDataset(kwargs['dev'], tokenizer, 1, dev_path, word2phonemes_path, **kwargs)
		test_dataset = LibriPhoneDataset(kwargs['test'], tokenizer, 1, test_path, word2phonemes_path, **kwargs)
		kwargs["ratio"] = 1.0
		kwargs["offset"] = 0
		train_dataset = LibriPhoneDataset(kwargs['train'], tokenizer, 1, train_path, word2phonemes_path, **kwargs)
	else:
		raise NotImplementedError

	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", vocab_size=tokenizer.vocab_size)

	config.adapter_name = args.trans_adapter_name
	config.output_adapter = args.output_adapter
	config.mh_adapter = args.mh_adapter
	config.prefix_tuning = args.prefix_tuning
	config.feat_enc_adapter = args.feat_enc_adapter
	config.lora_adapter = args.lora_adapter
	config.prefix_seq_len = args.prefix_seq_len
	config.prefix_projection = args.prefix_projection
	config.prefix_dropout_prob = args.prefix_dropout_prob
	config.ctc_loss_reduction = "mean"
	config.pad_token_id = tokenizer._vocab2idx["<pad>"]

	model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config, ignore_mismatched_sizes=True)

	model.freeze_feature_encoder()

	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))

	trainer = CustomTrainer(
		model=model,
		args=args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=dev_dataset,
		tokenizer=tokenizer,
		# callbacks = [TensorBoardCallback],
		callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
	)

	save_dir = join(args.output_dir, "best_model")
	if args.do_train:   # train and test
		trainer.train(resume_from_checkpoint=None)	  #join(args.output_dir, "checkpoint-4400")
		trainer.save_model(save_dir)

		test_metrics = trainer.predict(test_dataset).metrics
		print(test_metrics)

	if args.do_predict: # only for test
		device = trainer.model.device
		trainer.model = trainer.model.from_pretrained(save_dir).to(device)

		test_metrics = trainer.predict(test_dataset).metrics
		print(test_metrics)

if __name__ == "__main__":
	main()



