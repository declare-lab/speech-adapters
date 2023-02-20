import re
import numpy as np
import editdistance as ed

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def cer(hypothesis, groundtruth, **kwargs):
	err = 0
	tot = 0
	for p, t in zip(hypothesis, groundtruth):
		err += float(ed.eval(p, t))
		tot += len(t)
	return err / tot

def clean(ref):
	ref = re.sub(r'B\-(\S+) ', '', ref)
	ref = re.sub(r' E\-(\S+)', '', ref)
	return ref

def parse(hyp, ref):
	gex = re.compile(r'B\-(\S+) (.+?) E\-\1')

	# breakpoint()

	hyp = re.sub(r' +', ' ', hyp)
	ref = re.sub(r' +', ' ', ref)

	# breakpoint()

	hyp_slots = gex.findall(hyp)
	ref_slots = gex.findall(ref)

	ref_slots = ';'.join([':'.join([x[1], x[0]]) for x in ref_slots])
	if len(hyp_slots)>0:
		hyp_slots = ';'.join([':'.join([clean(x[1]), x[0]]) for x in hyp_slots])
	else:
		hyp_slots = ''

	ref = clean(ref)
	hyp = clean(hyp)

	# breakpoint()

	return ref, hyp, ref_slots, hyp_slots

#processor
tokenizer = Wav2Vec2CTCTokenizer("vocab_snips.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def compute_metric(pred):
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)

	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

	pred_str = processor.batch_decode(pred_ids)
	# we do not want to group tokens when computing the metrics
	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

	hypothesis = pred_str
	groundtruth = label_str

	value_hyps = []
	value_refs = []
	F1s = []
	for p, t in zip(hypothesis, groundtruth):
		ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
		ref_slots = ref_slots.split(';')
		hyp_slots = hyp_slots.split(';')
		unique_slots = []
		ref_dict = {}
		hyp_dict = {}
		if ref_slots[0] != '':
			for ref_slot in ref_slots:
				v, k = ref_slot.split(':')
				ref_dict.setdefault(k, [])
				ref_dict[k].append(v)
		if hyp_slots[0] != '':
			for hyp_slot in hyp_slots:
				v, k = hyp_slot.split(':')
				hyp_dict.setdefault(k, [])
				hyp_dict[k].append(v)
		# Slot Type F1 evaluation
		if len(hyp_dict.keys()) == 0 and len(ref_dict.keys()) == 0:
			F1 = 1.0
		elif len(hyp_dict.keys()) == 0:
			F1 = 0.0
		elif len(ref_dict.keys()) == 0:
			F1 = 0.0
		else:
			P, R = 0.0, 0.0
			for slot in ref_dict:
				if slot in hyp_dict:
					R += 1
			R = R / len(ref_dict.keys())
			for slot in hyp_dict:
				if slot in ref_dict:
					P += 1
			P = P / len(hyp_dict.keys())
			F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
		F1s.append(F1)
		# Slot Value WER/CER evaluation
		unique_slots = list(ref_dict.keys())
		for slot in unique_slots:
			for ref_i, ref_v in enumerate(ref_dict[slot]):
				if slot not in hyp_dict:
					hyp_v = ''
					value_refs.append(ref_v)
					value_hyps.append(hyp_v)
				else:
					min_cer = 100
					best_hyp_v = ""
					for hyp_v in hyp_dict[slot]:
						tmp_cer = cer([hyp_v], [ref_v])
						if min_cer > tmp_cer:
							min_cer = tmp_cer
							best_hyp_v = hyp_v
					value_refs.append(ref_v)
					value_hyps.append(best_hyp_v)

	return {"slot_type_f1" : sum(F1s) / len(F1s),
			"slot_value_cer" : cer(value_hyps, value_refs)}

def slot_type_f1_(hypothesis, groundtruth, **kwargs):
	F1s = []
	for p, t in zip(hypothesis, groundtruth):
		ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
		ref_slots = ref_slots.split(';')
		hyp_slots = hyp_slots.split(';')
		unique_slots = []
		ref_dict = {}
		hyp_dict = {}
		if ref_slots[0] != '':
			for ref_slot in ref_slots:
				v, k = ref_slot.split(':')
				ref_dict.setdefault(k, [])
				ref_dict[k].append(v)
		if hyp_slots[0] != '':
			for hyp_slot in hyp_slots:
				v, k = hyp_slot.split(':')
				hyp_dict.setdefault(k, [])
				hyp_dict[k].append(v)
		# Slot Type F1 evaluation
		if len(hyp_dict.keys()) == 0 and len(ref_dict.keys()) == 0:
			F1 = 1.0
		elif len(hyp_dict.keys()) == 0:
			F1 = 0.0
		elif len(ref_dict.keys()) == 0:
			F1 = 0.0
		else:
			P, R = 0.0, 0.0
			for slot in ref_dict:
				if slot in hyp_dict:
					R += 1
			R = R / len(ref_dict.keys())
			for slot in hyp_dict:
				if slot in ref_dict:
					P += 1
			P = P / len(hyp_dict.keys())
			F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
		F1s.append(F1)
	return sum(F1s) / len(F1s)

def slot_value_cer(hypothesis, groundtruth, **kwargs):
	value_hyps = []
	value_refs = []
	for p, t in zip(hypothesis, groundtruth):
		ref_text, hyp_text, ref_slots, hyp_slots = parse(p, t)
		ref_slots = ref_slots.split(';')
		hyp_slots = hyp_slots.split(';')
		unique_slots = []
		ref_dict = {}
		hyp_dict = {}
		if ref_slots[0] != '':
			for ref_slot in ref_slots:
				v, k = ref_slot.split(':')
				ref_dict.setdefault(k, [])
				ref_dict[k].append(v)
		if hyp_slots[0] != '':
			for hyp_slot in hyp_slots:
				v, k = hyp_slot.split(':')
				hyp_dict.setdefault(k, [])
				hyp_dict[k].append(v)
		# Slot Value WER/CER evaluation
		unique_slots = list(ref_dict.keys())
		for slot in unique_slots:
			for ref_i, ref_v in enumerate(ref_dict[slot]):
				if slot not in hyp_dict:
					hyp_v = ''
					value_refs.append(ref_v)
					value_hyps.append(hyp_v)
				else:
					min_cer = 100
					best_hyp_v = ""
					for hyp_v in hyp_dict[slot]:
						tmp_cer = cer([hyp_v], [ref_v])
						if min_cer > tmp_cer:
							min_cer = tmp_cer
							best_hyp_v = hyp_v
					value_refs.append(ref_v)
					value_hyps.append(best_hyp_v)

	return cer(value_hyps, value_refs)