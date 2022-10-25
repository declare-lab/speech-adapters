# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from torchaudio.datasets import LJSPEECH
from tqdm import tqdm

from fairseq.examples.speech_to_text.data_utils import save_df_to_tsv

from torch.utils.data import Dataset
import os
from os.path import join
import csv

import soundfile as sf


log = logging.getLogger(__name__)

SPLITS = ["train", "dev", "test"]

def get_tts_libritts(args):

	print("Generating manifest...")

	wrong_split = 0
	with open(join(args.output_data_root, "103_1241.trans.tsv")) as fd:
		rd = csv.reader(fd, delimiter="\t", quotechar='"')
		rows = []
		trans_audio_ids = []
		for row in rd:
			if len(row) != 3:
				wrong_split += 1
				new_row = [row[0], row[1].split("\t")[0], row[1].split("\t")[1]]
				row = new_row
			rows.append(row)
			trans_audio_ids.append(row[0])

	valid_len = int(len(trans_audio_ids) * 0.1)
	test_len = int(len(trans_audio_ids) * 0.1)

	train_ids = trans_audio_ids[:-(valid_len + test_len)]
	valid_ids = trans_audio_ids[-(valid_len + test_len):-test_len]
	test_ids = trans_audio_ids[-test_len:]

	id_to_split = {}
	for x in trans_audio_ids:
		if x in train_ids:
			id_to_split[x] = {}.get(x, "train")
		elif x in valid_ids:
			id_to_split[x] = {}.get(x, "dev")
		elif x in test_ids:
			id_to_split[x] = {}.get(x, "test")

	manifest_by_split = {split: defaultdict(list) for split in SPLITS}

	for i, row in enumerate(rows):
		# print(i, row)
		sample_id, utt, normalized_utt = row
		waveform, sample_rate = sf.read(join(args.output_data_root, sample_id+".wav"))
		split = id_to_split[sample_id]
		manifest_by_split[split]["id"].append(sample_id)
		audio_path = f"{args.output_data_root}/{sample_id}.wav"
		manifest_by_split[split]["audio"].append(audio_path)
		manifest_by_split[split]["n_frames"].append(len(waveform))
		manifest_by_split[split]["tgt_text"].append(normalized_utt)
		manifest_by_split[split]["speaker"].append("103")
		manifest_by_split[split]["src_text"].append(utt)

	manifest_root = Path(args.output_manifest_root).absolute()
	manifest_root.mkdir(parents=True, exist_ok=True)
	for split in SPLITS:
		save_df_to_tsv(
			pd.DataFrame.from_dict(manifest_by_split[split]),
			manifest_root / f"{split}.audio.tsv"
		)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-data-root", "-d", required=True, type=str)
	parser.add_argument("--output-manifest-root", "-m", required=True, type=str)
	args = parser.parse_args()

	get_tts_libritts(args)


if __name__ == "__main__":
	main()
