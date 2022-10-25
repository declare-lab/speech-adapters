import torch
import random
import numpy as np
import copy

from gc import callbacks
from transformers import set_seed, Wav2Vec2Processor
from transformers.integrations import TensorBoardCallback

from torch.utils.data import Dataset, DataLoader

import utils
from modules import CustomTrainer
from modeling_wav2vec2 import Wav2Vec2ForSequenceClassification
from data import EsdDataset, DataTrainingArguments, compute_metrics

def main():
	set_seed(1314)
	random.seed(100)

	args = DataTrainingArguments(
		output_dir = 'output',
		do_train = False,
		do_eval = True,
		do_predict = False,
		evaluation_strategy = "steps",
		save_strategy = "steps",
		save_steps=750,
		eval_steps=750,
		learning_rate=2e-5,
		per_device_train_batch_size=64,
		gradient_accumulation_steps=4,
		per_device_eval_batch_size=64,
		num_train_epochs=100,
		warmup_ratio=0.1,
		logging_steps=100,
		logging_dir='output/log',
		load_best_model_at_end=True,
		metric_for_best_model="accuracy",
	)

	emo_labels = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']
	num_labels = len(emo_labels)
	label2id, id2label = dict(), dict()
	for i, label in enumerate(emo_labels):
		label2id[label] = str(i)
		id2label[str(i)] = label

	test_list = utils.get_file_list(args.data_dir, mode="test")
	test_wav,  test_labels, sample_rate = utils.read_wav(test_list)

	# load pretrained model
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	model = Wav2Vec2ForSequenceClassification.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english",
			num_labels=num_labels,
			label2id=label2id,
			id2label=id2label)

	# audio dataset
	test_set = EsdDataset(test_wav, test_labels, processor, sample_rate)
	test_dataloader = DataLoader(test_set, batch_size=64)

	# freeze exclude prompt
	model.freeze_exclude_prompt()

	# model
	model = model.from_pretrained("output/best_model").cuda()
	model.eval()

	activation = {}
	def get_activation(name):
		def hook(model, input, output):
			activation[name] = output.detach()
		return hook

	print(model)
	activs_test = []
	model.wav2vec2.encoder.layers[23].promptblock.register_forward_hook(get_activation('act'))
	for inputs in test_dataloader:
		print(inputs.keys())
		input_values = inputs["input_values"].cuda()
		attention_mask = inputs["attention_mask"].cuda()
		labels = inputs.get("labels")[0].cuda()
		outputs = model(input_values, attention_mask, labels=labels)
		activs_test.extend(copy.deepcopy(activation['act']))

	#TSNE
	from sklearn.manifold import TSNE

	X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(np.array([i.cpu().numpy() for i in activs_test]))
	print(X_embedded.shape)


if __name__ == "__main__":
	main()
