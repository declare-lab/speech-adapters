import numpy as np
import torch

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, EarlyStoppingCallback, HfArgumentParser, Wav2Vec2FeatureExtractor
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
from modeling_wav2vec2 import Wav2Vec2ForXVector
from data import compute_metrics, get_sv_vctk_data

from transformers.adapters.prefix_tuning import PrefixTuningPool
from transformers.adapters import PrefixTuningConfig, PfeifferInvConfig

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

from torch import nn
class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		# forward pass
		outputs = model(**inputs)
		
		logits = outputs.get("logits")

		# compute custom loss (suppose one has 3 labels with different weights)
		# loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])).to(labels.device)  #add weight or not?
		loss_fct = nn.CrossEntropyLoss().to(labels.device)
		loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
		return (loss, outputs) if return_outputs else loss
	# def 



def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]

	#processor
	# processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	processor = feature_extractor

	# audio dataset
	# vctk
	if args.dataset.lower() == "vctk":
		train_set, train_list,_ = get_sv_vctk_data(args.data_dir, processor, "train")
		valid_set, valid_list,_ = get_sv_vctk_data(args.data_dir, processor, "evaluation")
		test_set,test_list,_= get_sv_vctk_data(args.data_dir, processor, "test")
	elif args.dataset.lower() == "voxceleb":
		pass
	else:
		raise NotImplementedError
	# voxceleb
	print("-------------Generating trails for Evaulation-----------------")
	classes = torch.ShortTensor(np.array(valid_set.labels))
	mask = classes.unsqueeze(1) == classes.unsqueeze(1).T
	tar_indices = torch.tril(mask, -1).numpy()
	non_indices = torch.tril(~mask, -1).numpy()

    # Select a subset of non-target trials to reduce the number of tests
	tar_non_ratio = np.sum(tar_indices)/np.sum(non_indices)
	non_indices *= (np.random.rand(*non_indices.shape) < tar_non_ratio)
	
	print(f"\t {np.sum(tar_indices)} target trials and")
	print(f"\t {np.sum(non_indices)} non-target trials")

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", #"anton-l/wav2vec2-base-superb-sv",
		num_labels=train_set.num_labels,
		label2id = train_set.label2id,
		id2label = train_set.id2label
	)

	
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
	# config.xvector_output_dim = train_set.num_labels


	# load pretrained model
	model = Wav2Vec2ForXVector.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	if args.prefixtuning:
		prefix_config = PrefixTuningConfig(flat=False, prefix_length=30)
		config.model_type = "wav2vec2"
		config.adapters.add("prefix_tuning", config=prefix_config)
		for module in model.modules():
			if isinstance(module, PrefixTuningPool):
				module.prefix_counts = {'prefix_tuning': {'self_prefix': 23}}
				module.confirm_prefix("prefix_tuning")
	
	# print("\n #Train: {}, #Valid: {}, #Test: {} ".format(len(train_set), len(valid_set), len(test_set)))

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))

	from torch.utils.data import DataLoader
	from datasets import load_metric

	# train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
	# test_dataloader = DataLoader(valid_set, batch_size=4, shuffle=True)
	# for inputs in train_dataloader:
	# 	labels = inputs["labels"]
	# 	outputs = model(**inputs)
	# 	print("=======================")
	# 	print(outputs)
	# 	print("=======================")
	# 	logits = outputs.get("logits")
	# 	print("labels:", labels)
	# 	print("logits:", logits.size())
	# 	metric = load_metric("accuracy")
	# 	new_test = labels.unsqueeze(1)
	# 	print("new_test:", new_test)
	# 	train = metric.compute(predictions=logits, references=new_test)
	# 	print(train)
	# 	exit()


	trainer = Trainer(
		model,
		args,
		train_dataset=train_set,
		eval_dataset=valid_set,
		tokenizer=processor,
		compute_metrics=compute_metrics,
		callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
	)

	save_dir = join(args.output_dir, "best_model")
	if args.do_train:   # train and test
		trainer.train(resume_from_checkpoint=None)    
		trainer.save_model(save_dir)
		# test_metrics = trainer.predict(test_set).metrics
		# print(test_metrics)

	if args.do_predict: # only for test
		device = trainer.model.device
		trainer.model = trainer.model.from_pretrained(save_dir).to(device)
		test_metrics= trainer.predict(test_set).metrics
		print(test_metrics)


if __name__ == "__main__":
	main()
