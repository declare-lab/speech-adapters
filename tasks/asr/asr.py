import numpy as np

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, EarlyStoppingCallback, HfArgumentParser
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
from modeling_wav2vec2 import Wav2Vec2ForCTC
from data import get_asr_data, DataCollatorCTCWithPadding, get_asr_meld_data, get_asr_fleurs_data, get_asr_voxpopuli_data
from datasets import load_metric

from transformers.trainer_utils import EvalPrediction

from transformers.adapters.prefix_tuning import PrefixTuningPool
from transformers.adapters import PrefixTuningConfig, PfeifferInvConfig


def make_compute_metrics(processor):
	def compute_metrics(pred):
		wer_metric = load_metric("wer")

		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)

		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

		pred_str = processor.batch_decode(pred_ids)
		# we do not want to group tokens when computing the metrics
		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

		wer = wer_metric.compute(predictions=pred_str, references=label_str)

		return {"wer": wer}
	return compute_metrics


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
	
	

def main():
	set_seed(1314)

	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]
	
	#processor
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

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
		train_set, max_len_train = get_asr_fleurs_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_asr_fleurs_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_asr_fleurs_data(args.data_dir, processor, "test")
	elif args.dataset.lower() == "voxpopuli":
		train_set, max_len_train = get_asr_voxpopuli_data(args.data_dir, processor, "train")
		valid_set, max_len_valid = get_asr_voxpopuli_data(args.data_dir, processor, "evaluation")
		test_set, max_len_test = get_asr_voxpopuli_data(args.data_dir, processor, "test")

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	
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
	model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	if args.prefixtuning:
		prefix_config = PrefixTuningConfig(flat=False, prefix_length=30)
		config.model_type = "wav2vec2"
		config.adapters.add("prefix_tuning", config=prefix_config)
		for module in model.modules():
			if isinstance(module, PrefixTuningPool):
				module.prefix_counts = {'prefix_tuning': {'self_prefix': 24}}
				module.confirm_prefix("prefix_tuning")
	
	print(model)

	print("\n #Train: {}, #Valid: {}, #Test: {} ".format(len(train_set), len(valid_set), len(test_set)))
	print(" #Train Max len: {}, #Valid Max len: {}, #Test Max len: {} \n".format(max_len_train, max_len_valid, max_len_test))

	## freeze all params exclude promptblock and classification head
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	if not args.fine_tune:
		model.freeze_exclude_prompt()
	print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.requires_grad, param.size())

	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=args,
		compute_metrics=make_compute_metrics(processor),
		train_dataset=train_set,
		eval_dataset=valid_set,
		tokenizer=processor.feature_extractor,
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


