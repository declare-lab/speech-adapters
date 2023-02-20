from datasets import load_dataset
from transformers import AdamW#, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments
import torch
import math

dataset = load_dataset("yelp_review_full", cache_dir='/data/yingting/Dataset/yelp_review_full')

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_constant_steps, num_training_steps, last_epoch=-1):
	def lr_lambda(current_step: int):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		elif current_step >= num_warmup_steps and  current_step < num_constant_steps:
			return float(1.0)
		return max(
			0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
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




tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="steps", 
					logging_steps=20, learning_rate=1e-2 , warmup_ratio=0.2, max_steps=1000, eval_steps=100)

trainer = CustomTrainer(
	model=model,
	args=training_args,
	train_dataset=small_train_dataset,
	eval_dataset=small_eval_dataset,
	compute_metrics=compute_metrics,
)

trainer.train()
