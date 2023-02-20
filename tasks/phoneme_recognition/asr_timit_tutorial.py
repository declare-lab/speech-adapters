import re
import json
from datasets import load_dataset, load_metric

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

def compute_metrics(pred):
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)

	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

	pred_str = processor.batch_decode(pred_ids)
	# we do not want to group tokens when computing the metrics
	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

	wer = wer_metric.compute(predictions=pred_str, references=label_str)

	return {"wer": wer}

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
	"""

	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# split inputs and labels since they have to be of different lenghts and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				return_tensors="pt",
			)

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch

timit = load_dataset("timit_asr", data_dir="/data/yingting/Dataset/TIMIT/")
print(timit)


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
	batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
	return batch

timit = timit.map(remove_special_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
	json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
	audio = batch["audio"]

	# batched output is "un-batched" to ensure mapping is correct
	batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
	batch["input_length"] = len(batch["input_values"])
	
	with processor.as_target_processor():
		batch["labels"] = processor(batch["text"]).input_ids
	return batch

timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)

max_input_length_in_sec = 4.0
timit["train"] = timit["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])



data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

model = Wav2Vec2ForCTC.from_pretrained(
	"facebook/wav2vec2-base",
	ctc_loss_reduction="mean", 
	pad_token_id=processor.tokenizer.pad_token_id,
)

repo_name = "wav2vec2-base-timit-demo-google-colab"

model.freeze_feature_encoder()

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
	model=model,
	data_collator=data_collator,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=timit["train"],
	eval_dataset=timit["test"],
	tokenizer=processor.feature_extractor,
)

train_dataloader = trainer.get_train_dataloader()
for inputs in train_dataloader:
  print(inputs["input_values"].size())
  print(inputs["labels"].size())
  print(inputs["input_values"])
  device = model.device
  input_values = inputs["input_values"].to(device)
  labels = inputs["labels"].to(device)
  outputs = model(input_values=input_values, labels=labels)
  logits = outputs.get("logits")
  print(logits.size())
  exit()

breakpoint()


