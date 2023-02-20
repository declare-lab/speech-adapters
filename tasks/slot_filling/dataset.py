import os
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio
from os.path import join
from pathlib import Path
import json 
import re
from os.path import exists
import numpy as np

from transformers import set_seed, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(txt):
	txt = re.sub(chars_to_ignore_regex, '', txt)#.lower()
	return txt	

def extract_all_chars(all_texts):
	all_text = " ".join(all_texts)
	vocab = list(set(all_text))
	return {"vocab": [vocab], "all_text": [all_text]}

class SnipsDataset(Dataset):
	def __init__(self, base_path, processor, mode):
		self.base_path = base_path
		self.processor = processor
		self.mode = mode

		self.datas = self.get_datas()#[:500]

	def get_datas(self):
		train_splits = []
		valid_splits = []
		test_splits = []
		vocab_snips = []
		all_text = []
		with open(join(self.base_path, "all.iob.snips.txt"), "r") as f:
			lines = f.readlines()
			for line in lines:
				splits = line.strip().split()
				utterence_id = splits[0]
				mode = utterence_id.split("-")[2]
				if mode == "train":
					train_splits.append(line)
				elif mode == "valid":
					valid_splits.append(line)
				else:
					test_splits.append(line)

				u_id_text, slot_names = line.strip().split("\t")
				u_id = u_id_text.split(" ")[0]
				text = u_id_text.split(" ")[1:][1:-1]
				clean_text = remove_special_characters(" ".join(text))
				slot = slot_names.split(" ")[1:-1]
				all_text.append(clean_text)
		if not os.path.exists("vocab.txt"):

			vocabs_snips = extract_all_chars(all_text)
			vocab_list = list(set(vocabs_snips["vocab"][0]))
			vocab_dict = {v: k for k, v in enumerate(vocab_list)}
			vocab_dict["|"] = vocab_dict[" "]
			del vocab_dict[" "]
			vocab_dict["[UNK]"] = len(vocab_dict)
			vocab_dict["[PAD]"] = len(vocab_dict)

			with open("vocab.txt", "w") as vf:
				for vl in vocab_list:
					vf.writelines(vl)

			with open('vocab.json', 'w') as vocab_file:
				json.dump(vocab_dict, vocab_file)

		if not os.path.exists("vocab_snips.json"):
			vocabs_snips = extract_all_chars(all_text)
			vocab_list = list(set(vocabs_snips["vocab"][0]))
			vocab_dict = {v: k for k, v in enumerate(vocab_list)}

			slots_file = join(self.base_path, "slots.txt")
			org_slots = open(slots_file).read().split('\n')
			slots = []
			for slot in org_slots[1:]:
				slots.append('B-'+slot)
				slots.append('E-'+slot)
			for slot in slots:
				vocab_dict[slot] = len(vocab_dict)

			vocab_dict["|"] = vocab_dict[" "]
			del vocab_dict[" "]
			vocab_dict["[UNK]"] = len(vocab_dict)
			vocab_dict["[PAD]"] = len(vocab_dict)

			with open('vocab_snips.json', 'w') as vocab_file:
				json.dump(vocab_dict, vocab_file)

		if self.mode == "train":
			return train_splits
		elif self.mode == "valid":
			return valid_splits
		else:
			return test_splits
		
	def __len__(self):
		return len(self.datas)

	def __getitem__(self, idx):
		u_id_text, slot_names = self.datas[idx].strip().split("\t")
		u_id = u_id_text.split(" ")[0]
		sent = u_id_text.split(" ")[1:][1:-1]
		iobs = slot_names.split(" ")[1:-1]

		processed_seqs = []
		for i, (wrd, iob) in enumerate(zip(sent, iobs)):
			if wrd in "?!.,;-":
				continue
			if wrd == '&':
				wrd = 'AND'
			if iob != 'O' and (i == 0 or iobs[i-1] != iob):
				processed_seqs.append('B-'+iob)
				processed_seqs.append("|")
			processed_seqs.append(wrd)
			if iob != 'O' and (i == len(sent)-1 or iobs[i+1] != iob):
				processed_seqs.append("|")
				processed_seqs.append('E-'+iob)
				processed_seqs.append("|")
			if i == (len(sent)-1):
				pass
			else:
				processed_seqs.append("|")

		# breakpoint()
		text_slot = self.processor.tokenizer.encode(" ".join(processed_seqs))

		wav_path = join(self.base_path, self.mode, u_id+".wav")
		wav, sr = torchaudio.load(wav_path)
		wav = wav.squeeze(0)
		input_value = self.processor(wav, sampling_rate=self.processor.feature_extractor.sampling_rate).input_values[0]

		# breakpoint()
		return {"input_values":input_value, 
				"labels":text_slot}

import abc
class _BaseTextEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, s):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, ids, ignore_repeat=False):
        raise NotImplementedError

    @abc.abstractproperty
    def vocab_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def token_type(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_from_file(cls, vocab_file):
        raise NotImplementedError

    @property
    def pad_idx(self):
        return 0

    @property
    def eos_idx(self):
        return 1

    @property
    def unk_idx(self):
        return 2

    def __repr__(self):
        return "<{} vocab_size={}>".format(type(self).__name__, self.vocab_size)

class CharacterTextSlotEncoder(_BaseTextEncoder):
    def __init__(self, vocab_list, slots):
        # Note that vocab_list must not contain <pad>, <eos> and <unk>
        # <pad>=0, <eos>=1, <unk>=2
        self._vocab_list = ["[PAD]", "|", "[UNK]"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}
        self.slots = slots
        self.slot2id = {self.slots[i]:(i+len(self._vocab_list)) for i in range(len(self.slots))}
        self.id2slot = {(i+len(self._vocab_list)):self.slots[i] for i in range(len(self.slots))}


    def encode(self, s):
        # Always strip trailing space, \r and \n
        sent, iobs = s.strip('\r\n ').split('\t')
        sent = sent.split(' ')[1:-1]
        iobs = iobs.split(' ')[1:-1]
        tokens = []
        for i, (wrd, iob) in enumerate(zip(sent, iobs)):
            if wrd in "?!.,;-":
                continue
            if wrd == '&':
                wrd = 'AND'
            if iob != 'O' and (i == 0 or iobs[i-1] != iob):
                tokens.append(self.slot2id['B-'+iob])
            tokens += [self.vocab_to_idx(v) for v in wrd]
            if iob != 'O' and (i == len(sent)-1 or iobs[i+1] != iob):
                tokens.append(self.slot2id['E-'+iob])
            if i == (len(sent)-1):
                tokens.append(self.eos_idx)
            else:
                tokens.append(self.vocab_to_idx(' '))
        return tokens

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            elif idx == self.eos_idx:
                break
            else:
                vocabs.append(v)
        return "".join(vocabs)

    @classmethod
    def load_from_file(cls, vocab_file, slots_file):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        org_slots = open(slots_file).read().split('\n')
        slots = []
        for slot in org_slots[1:]:
            slots.append('B-'+slot)
            slots.append('E-'+slot)
        return cls(vocab_list, slots)

    @property
    def vocab_size(self):
        return len(self._vocab_list) + len(self.slots)

    @property
    def token_type(self):
        return 'character-slot'

    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)

    def idx_to_vocab(self, idx):
        idx = int(idx)
        if idx < len(self._vocab_list):
            return self._vocab_list[idx]
        else:
            token = self.id2slot[idx]
            if token[0] == 'B':
                return token + ' '
            elif token[0] == 'E':
                return ' ' + token
            else:
                raise ValueError('id2slot get:', token)



if __name__ == "__main__":
	base_path = "/data/yingting/Dataset/SNIPS"

	#processor
	tokenizer = Wav2Vec2CTCTokenizer("vocab_snips.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	train_data = SnipsDataset(base_path, processor, "train")
	print("wav             :", train_data[0]["input_values"].shape)
	print("text_slot       :", train_data[0]["labels"])
	print("decode text_slot:", tokenizer.decode(train_data[0]["labels"]))
	# breakpoint()


		

