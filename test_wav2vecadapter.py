from transformers.adapters import BertAdapterModel, PfeifferInvConfig, PrefixTuningConfig, LoRAConfig

from transformers import (Wav2Vec2ForSequenceClassification, 
							Wav2Vec2Config, 
							Wav2Vec2Model, 
							Wav2Vec2Processor,
							Wav2Vec2PreTrainedModel
							)

from transformers.adapters import Wav2Vec2AdapterModel

import utils
from data import EsdDataset
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":
	emo_labels = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']
	num_labels = len(emo_labels)
	label2id, id2label = dict(), dict()
	for i, label in enumerate(emo_labels):
		label2id[label] = str(i)
		id2label[str(i)] = label


	# config.add_adapter
	# config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

	config = Wav2Vec2Config.from_pretrained(
		"jonatasgrosman/wav2vec2-large-xlsr-53-english",
		num_labels=5,
		# add_adapter=True,
		label2id = label2id,
		id2label = id2label
	)
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

	test_list = utils.get_file_list("/data/yingting/ESD/en/", mode="test")
	test_wav,  test_labels, sample_rate = utils.read_wav(test_list)
	test_set = EsdDataset(test_wav, test_labels, processor, sample_rate)
	test_loader = DataLoader(test_set, batch_size=64)

	model = Wav2Vec2AdapterModel(config=config)
	# model = Wav2Vec2AdapterModel.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)

	for inputs in test_loader:
		output = model(inputs.input_values)

	print(model)

