import numpy as np

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, HfArgumentParser, EarlyStoppingCallback
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
from modules import CustomTrainer
from modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
from data import get_sp_cls_data, compute_metrics, VCTKMultiSpkDataset, get_sp_vctk_data

from transformers.adapters.prefix_tuning import PrefixTuningPool
from transformers.adapters import PrefixTuningConfig, PfeifferInvConfig

from datasets import load_dataset	

@dataclass
class DataTrainingArguments(TrainingArguments):
	data_dir: Optional[str] = field(
		default="/data/yingting/VCTK/", metadata={"help": "The dir of the dataset."}
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
	
	

def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]
	
	#processor
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	# audio dataset
	### VCTK
	# print("=========>>>>> Loading the vctk dataset")
	# "/data/yingting/VCTK/downloads/extracted/data/wav48_silence_trimmed"
	# train_set = VCTKMultiSpkDataset(args.data_dir, processor, 16000, cv=0)
	# valid_set = VCTKMultiSpkDataset(args.data_dir, processor, 16000, cv=1)
	# test_set = VCTKMultiSpkDataset(args.data_dir, processor, 16000, cv=2)

	### ESD
	# "/data/yingting/ESD/en/"
	# train_set, max_len_train = get_sp_cls_data(args.data_dir, processor, "train")
	# valid_set, max_len_valid = get_sp_cls_data(args.data_dir, processor, "evaluation")
	# test_set, max_len_test = get_sp_cls_data(args.data_dir, processor, "test")

	### VCTK wav
	# "/data/yingting/VCTK_Wav/wav48/"

	print("============================")
	print("------>>>>>> train")
	train_set, _ = get_sp_vctk_data(args.data_dir, processor, "train")
	print("------>>>>>> valid")
	valid_set, _ = get_sp_vctk_data(args.data_dir, processor, "evaluation")
	print("------>>>>>> test")
	test_set, _ = get_sp_vctk_data(args.data_dir, processor, "test")

	
	print("len of train_set:", len(train_set))
	print("len of valid_set:", len(valid_set))
	print("len of test_set:", len(test_set))

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english",
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


	# load pretrained model
	model = Wav2Vec2ForSequenceClassification.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	if args.prefixtuning:
		prefix_config = PrefixTuningConfig(flat=False, prefix_length=30)
		config.model_type = "wav2vec2"
		config.adapters.add("prefix_tuning", config=prefix_config)
		for module in model.modules():
			if isinstance(module, PrefixTuningPool):
				module.prefix_counts = {'prefix_tuning': {'self_prefix': 23}}
				module.confirm_prefix("prefix_tuning")
	
	print(model)

	print("\n #Train: {}, #Valid: {}, #Test: {} \n".format(len(train_set), len(valid_set), len(test_set)))
	# print(" #Train Max len: {}, #Valid Max len: {}, #Test Max len: {} \n".format(max_len_train, max_len_valid, max_len_test))

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.requires_grad, param.size())

	trainer = CustomTrainer(
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
		test_metrics = trainer.predict(test_set).metrics
		print(test_metrics)

	if args.do_predict: # only for test
		device = trainer.model.device
		trainer.model = trainer.model.from_pretrained(save_dir).to(device)
		test_metrics= trainer.predict(test_set).metrics
		print(test_metrics)
	

if __name__ == "__main__":
	main()

