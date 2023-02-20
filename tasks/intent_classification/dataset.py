from torch.utils.data import Dataset
from os.path import join
import csv
import librosa
from pathlib import Path
from transformers import Wav2Vec2Processor
import random
random.seed(4)

class ICDataset(Dataset):
	def __init__(self, root_dir, mode, processor: Wav2Vec2Processor):

		self.processor = processor
		if mode=="train":
			file_name = "train_data.csv"
		elif mode=="valid":
			file_name = "valid_data.csv"
		elif mode=="test":
			file_name = "test_data.csv"
		else:
			raise "mode need be one of {train, valid, test}"
		csv_file =  join(root_dir, "data", file_name)

		samples = []

		with open(csv_file, newline='') as csvfile:
			reader = csv.reader(csvfile)
			line_count = 0
			for i, row in enumerate(reader):
				if line_count == 0:
					line_count += 1
				else:
					line_count += 1
					samples.append({
						"path":join(root_dir, row[1]),
						"speaker_id":row[2],
						"text":row[3],
						"action":row[4],
						"object":row[5],
						"location":row[6]
					})
		# Shuffle
		if mode == "train":
			random.shuffle(samples)

		self.id2label = {0: 'change language', 1: 'activate', 2: 'deactivate', 3: 'increase', 4: 'decrease', 
						 5: 'bring', 6: 'none_object', 7: 'music', 8: 'lights', 9: 'volume', 10: 'heat', 
						 11: 'lamp', 12: 'newspaper', 13: 'juice', 14: 'socks', 15: 'Chinese', 16: 'Korean', 
						 17: 'English', 18: 'German', 19: 'shoes', 20: 'none_location', 21: 'kitchen', 
						 22: 'bedroom', 23: 'washroom'}
		self.label2id = {v:k for k,v in self.id2label.items()}
		self.Sy_intent = {'action': {'change language': 0, 0: 'change language', 'activate': 1, 1: 'activate', 
									'deactivate': 2, 2: 'deactivate', 'increase': 3, 3: 'increase', 
									'decrease': 4, 4: 'decrease', 'bring': 5, 5: 'bring'}, 
						  'object': {'none': 0, 0: 'none', 'music': 1, 1: 'music', 'lights': 2, 2: 'lights', 
						  			'volume': 3, 3: 'volume', 'heat': 4, 4: 'heat', 'lamp': 5, 5: 'lamp', 
									'newspaper': 6, 6: 'newspaper', 'juice': 7, 7: 'juice', 'socks': 8, 8: 'socks', 
									'Chinese': 9, 9: 'Chinese', 'Korean': 10, 10: 'Korean', 'English': 11, 11: 'English', 
									'German': 12, 12: 'German', 'shoes': 13, 13: 'shoes'}, 
						  'location': {'none': 0, 0: 'none', 'kitchen': 1, 1: 'kitchen', 
						  			'bedroom': 2, 2: 'bedroom', 'washroom': 3, 3: 'washroom'}}
		
		self.samples = samples
		self.num_labels = len(self.id2label)

	def __getitem__(self, index):

		audio_wav, _ = librosa.load(self.samples[index]["path"], sr=16000, mono=True)

		inputs = self.processor(audio_wav, padding="max_length", max_length=220000, truncation=True, sampling_rate=16000, return_tensors="pt")

		label = []
		for slot in ["action", "object", "location"]:
			value = self.samples[index][slot]
			label.append(self.Sy_intent[slot][value])
		
		return {'input_values'   : inputs.input_values.squeeze(0), 
				'attention_mask' : inputs.attention_mask.squeeze(0),
				'labels': label
				# 'intents': label
				}
	
	def __len__(self):
		return len(self.samples)

if __name__=="__main__":
	path = "/data/yingting/Dataset/fluent_speech_commands_dataset/"
	processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
	icdataset = ICDataset(path, "train", processor)
	print("len of icdataset:", len(icdataset))
	print(icdataset[0])