# Set up
## Install flashlight
using same conda envs with before, and install the following packages:

follow [this github](https://github.com/flashlight/flashlight/tree/main/bindings/python#dependencies) to install the Dependencies first

```python
pip install flashlight==1.0.0
```

## Data download

Download the data from [here](https://openslr.org/60/) to "libritts/", we are using "train-clean-100.tar.gz", and then use cmd "tar zxvf train-clean-100.tar.gz"

## Download the checkpoints

1. Download the wav2vec2-checkpoint from  [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) and put under the "libritts/wav2vec2_checkpoint/" directory, we are using the "Wav2Vec 2.0 Base	10 minutes Librispeech" line.

2. Download the "dict.ltr.txt" of librispeech and put under "libritts/wav2vec2_checkpoint/" directory

```python 
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
```

3. Download the pretrained_model vctk_transformer_phn checkpoints from [here](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_synthesis/docs/vctk_example.md) to "libritts/pretrained_model/vctk_transformer_phn/"


# Data process
```python
bash data_preprocess.sh
```

## mv related data to the correct dir

~~1. cp "gcmvn_stats.npz" from "libritts/feature_manifest_root_vctk/" to "~/PromptSpeech/tasks/tts_vctk/"~~
```python
cp gcmvn_stats.npz ~/PromptSpeech/tasks/tts_vctk/
```

~~2. cp "train.tsv test.tsv dev.tsv" from "libritts/feature_manifest_root_vctk/" to "libritts/pretrained_model/vctk_transformer_phn/"~~
```python
cd libritts/feature_manifest_root_vctk/
cp train.tsv test.tsv dev.tsv ../pretrained_model/vctk_transformer_phn/
```
3. cp "spm_char.txt" from pretrained_model to feature_manifest_root_vctk
```python
cd libritts/feature_manifest_root_vctk/
mv spm_char.txt backup.spm_char.txt
cd libritts/pretrained_model/vctk_transformer_phn/
cp spm_char.txt ../../feature_manifest_root_vctk/
```


# Train
```python
bash tts_train.sh
```

# Inference
```python
bash tts_inference.sh
```

