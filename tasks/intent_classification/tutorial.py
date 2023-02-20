import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "ic", split="test")

print("====================================")
print(dataset)

print("\n")
print(dataset[0])
print("\n")

dataset = dataset.map(map_to_array)

model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-ic")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ic")

print("\n-----------------------")
print(model.config.id2label)
print("------------------------\n")

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
breakpoint()

logits = model(**inputs).logits

print("logits:", logits.size())
exit()

action_ids = torch.argmax(logits[:, :6], dim=-1).tolist()
action_labels = [model.config.id2label[_id] for _id in action_ids]

object_ids = torch.argmax(logits[:, 6:20], dim=-1).tolist()
object_labels = [model.config.id2label[_id + 6] for _id in object_ids]

location_ids = torch.argmax(logits[:, 20:24], dim=-1).tolist()
location_labels = [model.config.id2label[_id + 20] for _id in location_ids]
