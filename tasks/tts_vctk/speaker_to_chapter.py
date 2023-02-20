import torch
import numpy as np
import json


speaker_emb_path = "/data/yingting/libritts/speaker.pkl"
speaker_emb_dict = np.load(speaker_emb_path, allow_pickle=True)

speaker_to_chapter = {}

for k, v in speaker_emb_dict.items():
	k_list = k.split("_")
	speaker_id = k_list[0]
	chapter_id = k_list[1]
	
	if speaker_id not in speaker_to_chapter.keys():
		speaker_to_chapter[speaker_id] = set()
		speaker_to_chapter[speaker_id].add(chapter_id)
	else:
		speaker_to_chapter[speaker_id].add(chapter_id)

for k in speaker_to_chapter.keys():
	speaker_to_chapter[k] = list(speaker_to_chapter[k])

with open('speaker_to_chapter.json', 'w') as fp:
    json.dump(speaker_to_chapter, fp)




	
