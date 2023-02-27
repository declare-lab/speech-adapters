import numpy as np

from dataclasses import field, dataclass
from typing import *
from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, HfArgumentParser, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from os.path import join

set_seed(1314)

from path import Path
import sys
folder = Path(__file__).abspath()
sys.path.append(folder.parent)
sys.path.append(folder.parent.parent.parent)

from dataset import ICDataset
from customtrain import CustomTrainer, compute_metrics
from modeling_wav2vec2 import Wav2Vec2ForSequenceClassification


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
	
	# args
	parser = HfArgumentParser(DataTrainingArguments)
	args = parser.parse_args_into_dataclasses()[0]
	
	#processor
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	#dataset  fluent_commands
	train_set = ICDataset(args.data_dir, "train", processor)
	valid_set = ICDataset(args.data_dir, "valid", processor)
	test_set = ICDataset(args.data_dir, "test", processor)

	# breakpoint()

	# config
	config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english",
		num_labels=train_set.num_labels,
		label2id = train_set.label2id,
		id2label = train_set.id2label
	)
	
	config.adapter_name = args.trans_adapter_name
	config.output_adapter = args.output_adapter
	config.mh_adapter = args.mh_adapter
	# config.prefixtuning = args.prefixtuning
	config.prefix_tuning = args.prefix_tuning
	config.feat_enc_adapter = args.feat_enc_adapter
	config.lora_adapter = args.lora_adapter
	config.prefix_seq_len = args.prefix_seq_len
	config.prefix_projection = args.prefix_projection
	config.prefix_dropout_prob = args.prefix_dropout_prob


	# load pretrained model
	model = Wav2Vec2ForSequenceClassification.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
	
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
	# breakpoint()

	save_dir = join(args.output_dir, "best_model")
	if args.do_train:   # train and test
		trainer.train(resume_from_checkpoint=None)    #join(args.output_dir, "checkpoint-4000")
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