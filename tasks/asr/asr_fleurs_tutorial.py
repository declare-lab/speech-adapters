from datasets import load_dataset, load_metric
import numpy as np

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor


@dataclass
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

	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True
	max_length: Optional[int] = None
	max_length_labels: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	pad_to_multiple_of_labels: Optional[int] = None
	
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

fleurs = load_dataset("google/xtreme_s", "fleurs.en_us", cache_dir="/data/yingting/fleurs")
print(fleurs)

fleurs = fleurs.remove_columns(["num_samples", "raw_transcription", "gender", "lang_id","language", "lang_group_id"])

import random

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

	# for idx in picks:
	# 	print(processor.batch_decode(dataset[idx]["labels"]))

print(len(fleurs["train"]))
show_random_elements(fleurs["train"].remove_columns(['path', 'audio']))

import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
	batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower()
	return batch

fleurs = fleurs.map(remove_special_characters)

# def extract_all_chars(batch):
#   all_text = " ".join(batch["transcription"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}

# vocabs = fleurs.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=fleurs.column_names["train"])
# vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

# vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)

# print(vocab_dict)

# import json
# with open('vocab_fleurs.json', 'w') as vocab_file:
# 	json.dump(vocab_dict, vocab_file)


tokenizer = Wav2Vec2CTCTokenizer("./vocab_fleurs.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
	audio = batch["audio"]

	# batched output is "un-batched" to ensure mapping is correct
	batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
	batch["input_length"] = len(batch["input_values"])
	
	with processor.as_target_processor():
		batch["labels"] = processor(batch["transcription"]).input_ids
	return batch

fleurs = fleurs.map(prepare_dataset, remove_columns=fleurs.column_names["train"], num_proc=4)

max_input_length_in_sec = 20.0
fleurs = fleurs.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
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

model = Wav2Vec2ForCTC.from_pretrained(
	"jonatasgrosman/wav2vec2-large-xlsr-53-english", 
	ctc_loss_reduction="mean", 
	pad_token_id=processor.tokenizer.pad_token_id,
	vocab_size=len(processor.tokenizer),
	ignore_mismatched_sizes=True
)
model.freeze_feature_encoder()

training_args = TrainingArguments(
  output_dir="/data/yingting/output_asr_fleurs_use_fleursvocab/",
  group_by_length=True,
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True, 
  save_steps=100,
  eval_steps=100,
  logging_steps=100,
  learning_rate=5e-3,
  weight_decay=0.005,
  warmup_steps=1000,
#   save_total_limit=2,
)

# --load_best_model_at_end True 

trainer = Trainer(
	model=model,
	data_collator=data_collator,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=fleurs["train"],
	eval_dataset=fleurs["validation"],
	tokenizer=processor.feature_extractor,
)
trainer.train(resume_from_checkpoint=None)

# processor = Wav2Vec2Processor.from_pretrained("/data/yingting/output_asr_fleurs/checkpoint-500/")
# model = Wav2Vec2ForCTC.from_pretrained("/data/yingting/output_asr_fleurs/checkpoint-1200/").cuda()

def map_to_result(batch):
	with torch.no_grad():
		input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0).to(model.device)
		logits = model(input_values).logits

	print("logits:", logits.size())
	pred_ids = torch.argmax(logits, dim=-1)
	print("pred_ids:", pred_ids)
	batch["pred_str"] = processor.batch_decode(pred_ids)[0]
	batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
	return batch

results = fleurs["test"].map(map_to_result, remove_columns=fleurs["test"].column_names)

for idx in range(len(results)):
	print("pred_str:", results[idx]["pred_str"])
	print("text:", results[idx]["text"])

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))




