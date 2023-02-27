# ICASSP2023 EVALUATING PARAMETER-EFFICIENT TRANSFER LEARNING APPROACHES ON SURE BENCHMARK FOR SPEECH UNDERSTANDING

## Set up
```python
conda create --name speechprompt python==3.8.5
conda activate speechprompt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
pip install -r requirements.txt
```

## Dataset

### ESD Dataset
download [link](https://github.com/HLTSingapore/Emotional-Speech-Data)

### VCTK Dataset
1. load from huggingface ```load_dataset("vctk", split='train', cache_dir='/data/path/VCTK')```
2. or can download raw data from [link](https://datashare.ed.ac.uk/handle/10283/2651) and follow the data preparation strategy of [nuwave](https://github.com/mindslab-ai/nuwave)
### Google Speech Commands Dataset
download [link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
, also can use this [link](https://github.com/NVIDIA/NeMo/blob/v0.10.1/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb) to download

only use these ['off', 'up', 'stop', 'four', 'no', 'down', 'left', 'go', 'yes', 'on', 'right'] classes, same with [this](https://arxiv.org/ftp/arxiv/papers/2101/2101.04792.pdf) paper 

### VoxCeleb1 Dataset
reference [this](https://github.com/clovaai/voxceleb_trainer)


## Train
Currently our benchmark includes tasks such as emotion cls, asr, intent cls, keyword_spotting, phoneme recognition, slot filling, speaker cls, tts. For each task, we implemented fine-tune, prefix-tuning, lora, bottleneck adapter, and convadapter which is proposed in the paper.

For example, start a new emotion classification task, we will set the corresponding parameter like below:
```python
## finetune
--fine_tune True
## bottleneck
--trans_adapter_name "bottleneck"
--output_adapter True
## prefix-tuning
--prefix_tuning True
## lora
--lora_adapter True
## adapterblock
--trans_adapter_name "adapterblock"
--output_adapter True
```

We also examples in emotion_cls.sh, use this command to start new emotion classification task:
```python
bash emotion_cls.sh
```

For tts task:
We use [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS), choice "transformer" as the backbone model, and implement fine-tune, prefix-tuning, lora, bottleneck adapter, and convadapter on it.


## Tensorboard
```python
tensorboard --logdir=/data/path/output_earlystop_asr_fleurs_lora_2e3/log --bind_all
```
