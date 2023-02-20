import numpy as np
import torch
import math

from dataclasses import field, dataclass
from datasets import load_dataset
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Tokenizer
from transformers import TrainingArguments, EarlyStoppingCallback, HfArgumentParser, Trainer
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

from dataset import SnipsDataset
from data import DataCollatorCTCWithPadding
from metric import compute_metric
from modeling_wav2vec2 import Wav2Vec2ForCTC

from torch.optim.lr_scheduler import LambdaLR

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

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_constant_steps, num_training_steps, last_epoch=-1):
	def lr_lambda(current_step: int):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		elif current_step >= num_warmup_steps and  current_step < num_constant_steps:
			return float(1.0)
		return max(
			0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_constant_steps))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomTrainer(Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.constant_ratio = 0.4
		self.num_constant_steps = -1
	def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
		if self.lr_scheduler is None:
			self.lr_scheduler = get_linear_schedule_with_warmup(
				self.optimizer if optimizer is None else optimizer, 
				num_warmup_steps=self.args.get_warmup_steps(num_training_steps), 
				num_constant_steps=self.get_keep_constant_steps(num_training_steps), 
				num_training_steps=num_training_steps)
		return self.lr_scheduler
	def get_keep_constant_steps(self, num_training_steps: int):
		keep_constant_steps = (
			self.num_constant_steps if self.num_constant_steps > 0 else math.ceil(num_training_steps * (self.constant_ratio + self.args.warmup_ratio))
		)
		return keep_constant_steps

def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]

	#processor
	tokenizer = Wav2Vec2CTCTokenizer("vocab_snips.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	# audio dataset
	# snips
	if args.dataset.lower() == "snips":
		train_set = SnipsDataset(args.data_dir, processor, "train")
		valid_set = SnipsDataset(args.data_dir, processor, "valid")
		test_set = SnipsDataset(args.data_dir, processor, "test")
	elif args.dataset.lower() == "voxceleb":
		pass
	else:
		raise NotImplementedError

	print("====================")
	print("len of train:", len(train_set))
	print("len of valid:", len(valid_set))
	print("len of test :", len(test_set))

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", vocab_size=len(processor.tokenizer))
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
	model.freeze_feature_encoder()

	print("\n #Train: {}, #Valid: {}, #Test: {} ".format(len(train_set), len(valid_set), len(test_set)))

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	# for name, param in model.named_parameters():
	# 	if param.requires_grad:
	# 		print(name, param.requires_grad, param.size())

	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, max_length=200000)

	trainer = CustomTrainer(
		model=model,
		data_collator=data_collator,
		args=args,
		compute_metrics=compute_metric,
		train_dataset=train_set,
		eval_dataset=valid_set,
		tokenizer=processor.tokenizer,
		# callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
		callbacks = [TensorBoardCallback],
	)

	save_dir = join(args.output_dir, "best_model")
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