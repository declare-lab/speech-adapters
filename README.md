# Evaluating parameter-efficient transfer learning approaches on SURE benchmark for speech understanding

## [Paper](https://arxiv.org/pdf/2303.03267.pdf)

Fine-tuning is widely used as the default algorithm for transfer learning from pre-trained models. Parameter inefficiency can however arise when, during transfer learning, all the parameters of a large pre-trained model need to be updated for individual downstream tasks. As the number of parameters grows, fine-tuning is prone to overfitting and catastrophic forgetting. In addition, full fine-tuning can become prohibitively expensive when the model is used for many tasks. To mitigate this issue, parameter-efficient transfer learning algorithms, such as adapters and prefix tuning, have been proposed as a way to introduce a few trainable parameters that can be plugged into large pre-trained language models such as BERT, and HuBERT. In this paper, we introduce the Speech UndeRstanding Evaluation (SURE) benchmark for parameter-efficient learning for various speech-processing tasks. Additionally, we introduce a new adapter, ConvAdapter, based on 1D convolution. We show that ConvAdapter outperforms the standard adapters while showing comparable performance against prefix tuning and LoRA with only 0.94% of trainable parameters on some of the tasks in SURE. We further explore the effectiveness of parameter efficient transfer learning for speech synthesis task such as Text-to-Speech (TTS).

## Set up
```python
conda create --name speechprompt python==3.8.5
conda activate speechprompt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
pip install -r requirements.txt
```

## Dataset

![image](https://user-images.githubusercontent.com/35062414/221520253-3fba52bf-ff2f-4a2a-8199-be75d4de3989.png)


### ESD Dataset
Here is the download [link](https://github.com/HLTSingapore/Emotional-Speech-Data)

### VCTK Dataset
1. load from huggingface ```load_dataset("vctk", split='train', cache_dir='/data/path/VCTK')```
2. or can download raw data from [link](https://datashare.ed.ac.uk/handle/10283/2651) and follow the data preparation strategy of [nuwave](https://github.com/mindslab-ai/nuwave)
### Google Speech Commands Dataset
Here is the download [link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
, also can use this [link](https://github.com/NVIDIA/NeMo/blob/v0.10.1/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb) to download.

Only use these ['off', 'up', 'stop', 'four', 'no', 'down', 'left', 'go', 'yes', 'on', 'right'] classes, same as in [this](https://arxiv.org/ftp/arxiv/papers/2101/2101.04792.pdf) paper 

### VoxCeleb1 Dataset
Download it from [this](https://github.com/clovaai/voxceleb_trainer)


## Train
Currently our benchmark includes tasks such as emotion cls, asr, intent cls, keyword_spotting, phoneme recognition, slot filling, speaker cls, tts. For each task, we implemented fine-tune, prefix-tuning, lora, bottleneck adapter, and convadapter which is proposed in the paper.

![image](https://user-images.githubusercontent.com/35062414/221511052-a6f4c44a-f779-4fca-9142-6ea10254b764.png)

![image](https://user-images.githubusercontent.com/35062414/221511119-27c65410-3086-4509-8927-1ce43efc13af.png)


### Emotion classification
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

### Tts
We use [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS), choice "transformer" as the backbone model, and implement fine-tune, prefix-tuning, lora, bottleneck adapter, and convadapter on it.


## Tensorboard
```python
tensorboard --logdir=/data/path/output_earlystop_asr_fleurs_lora_2e3/log --bind_all
```
