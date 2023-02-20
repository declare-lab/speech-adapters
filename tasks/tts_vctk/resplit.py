import torch
import csv
import random
import pandas as pd

train_valid = "/data/yingting/libritts/feature_manifest_root_vctk/train_valid.tsv"
 
samples = pd.read_csv(train_valid, sep='\t')

num_samples = len(samples["id"])

print("num_samples:", num_samples)

train_ids = []
valid_ids = []
speaker2ids = {}

for i in range(num_samples):
	speaker_id = samples["speaker"][i]
	if speaker_id not in speaker2ids.keys():
		speaker2ids[speaker_id] = [samples["id"][i]]
	else:
		speaker2ids[speaker_id].append(samples["id"][i])

print("len of speaker2ids: ", len(speaker2ids.keys()))

for key in speaker2ids.keys():
	num_key_samples = len(speaker2ids[key])
	sample_ids = speaker2ids[key]
	random.seed(4)
	random.shuffle(sample_ids)
	num4train = int(num_key_samples * 0.9)
	train_ids.append(sample_ids[:num4train])
	valid_ids.append(sample_ids[num4train:])

	# print(key, " len of samples:", len(speaker2ids[key]), " for train: ", len(sample_ids[:num4train]), " for valid: ", len(sample_ids[num4train:]))

ids4train = [i for k in train_ids for i in k]
ids4valid = [i for k in valid_ids for i in k]

print("len of ids4train:", len(ids4train))
print("len of ids4valid:", len(ids4valid))


with open("/data/yingting/libritts/feature_manifest_root_vctk/train.tsv", "w", encoding='utf8', newline='') as tsv_file:
	tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
	tsv_writer.writerow(["id", "audio", "n_frames", "tgt_text", "speaker", "src_text"])

	for pos, id_ in enumerate(samples["id"]):
		if id_ in ids4train:
			tsv_writer.writerow([samples["id"][pos], samples["audio"][pos], samples["n_frames"][pos], samples["tgt_text"][pos], samples["speaker"][pos], samples["src_text"][pos]])

with open("/data/yingting/libritts/feature_manifest_root_vctk/dev.tsv", "w", encoding='utf8', newline='') as tsv_file:
	tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
	tsv_writer.writerow(["id", "audio", "n_frames", "tgt_text", "speaker", "src_text"])

	for pos, id_ in enumerate(samples["id"]):
		if id_ in ids4valid:
			tsv_writer.writerow([samples["id"][pos], samples["audio"][pos], samples["n_frames"][pos], samples["tgt_text"][pos], samples["speaker"][pos], samples["src_text"][pos]])


