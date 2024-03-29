import numpy as np

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, EarlyStoppingCallback, HfArgumentParser
from transformers.integrations import TensorBoardCallback

import os
from os.path import join
import utils
from modules import CustomTrainer
from modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
from data import get_data, compute_metrics, get_emo_cls_iemocap_data, get_emo_meld_data, compute_metrics_macro_f1

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
	
	

def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]
	
	#processor
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	# audio dataset
	if args.dataset.lower() == "esd":
		### ESD
		train_set, max_len_train = get_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_data(args.data_dir, processor, "test")

		# train_set = get_emo_cls_data(args.data_dir, processor, "train")  # for read wav when need
		# valid_set = get_emo_cls_data(args.data_dir, processor, "evaluation")
		# test_set  = get_emo_cls_data(args.data_dir, processor, "test")
	elif args.dataset.lower() == "iemocap":
		### IEMOCAP
		wav_file_names, emotions = utils.get_iemocap_labels(args.data_dir)

		train_set, max_len_train = get_emo_cls_iemocap_data(args.data_dir, processor, "train", wav_file_names, emotions)
		valid_set, max_len_valid = get_emo_cls_iemocap_data(args.data_dir, processor, "evaluation", wav_file_names, emotions)
		test_set, max_len_test = get_emo_cls_iemocap_data(args.data_dir, processor, "test", wav_file_names, emotions)

	elif args.dataset.lower() == "meld":
		### MELD
		train_set, max_len_train = get_emo_meld_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_emo_meld_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_emo_meld_data(args.data_dir, processor, "test")

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english",
		num_labels=train_set.num_labels,
		label2id = train_set.label2id,
		id2label = train_set.id2label
	)
	
	config.adapter_name = args.trans_adapter_name
	config.output_adapter = args.output_adapter
	config.mh_adapter = args.mh_adapter
	config.prefix_tuning = args.prefix_tuning
	config.feat_enc_adapter = args.feat_enc_adapter
	config.lora_adapter = args.lora_adapter
	config.prefix_seq_len = args.prefix_seq_len
	config.prefix_projection = args.prefix_projection
	config.prefix_dropout_prob = args.prefix_dropout_prob


	# load pretrained model
	model = Wav2Vec2ForSequenceClassification.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	
	print(model)

	print("\n #Train: {}, #Valid: {}, #Test: {} ".format(len(train_set), len(valid_set), len(test_set)))
	# print(" #Train Max len: {}, #Valid Max len: {}, #Test Max len: {} \n".format(max_len_train, max_len_valid, max_len_test))
	print("\n")

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.requires_grad, param.size())

	if args.metric_for_best_model == "f1":
		com_metrics = compute_metrics_macro_f1
	else:
		com_metrics = compute_metrics


	trainer = CustomTrainer(
		model,
		args,
		train_dataset=train_set,
		eval_dataset=valid_set,
		tokenizer=processor,
		# compute_metrics=compute_metrics,
		compute_metrics=com_metrics,
		callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
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


