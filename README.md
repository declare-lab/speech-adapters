# Evaluating parameter-efficient transfer learning approaches on SURE benchmark for speech understanding

## [Paper](https://arxiv.org/pdf/2303.03267.pdf)

## Motivation

Fine-tuning is widely used as the default algorithm for transfer learning from pre-trained models. Parameter inefficiency can however arise when, during transfer learning, all the parameters of a large pre-trained model need to be updated for individual downstream tasks. As the number of parameters grows, fine-tuning is prone to overfitting and catastrophic forgetting. In addition, full fine-tuning can become prohibitively expensive when the model is used for many tasks. To mitigate this issue, parameter-efficient transfer learning algorithms, such as adapters and prefix tuning, have been proposed as a way to introduce a few trainable parameters that can be plugged into large pre-trained language models such as BERT, and HuBERT. In this paper, we introduce the Speech UndeRstanding Evaluation (SURE) benchmark for parameter-efficient learning for various speech-processing tasks. Additionally, we introduce a new adapter, ConvAdapter, based on 1D convolution. We show that ConvAdapter outperforms the standard adapters while showing comparable performance against prefix tuning and LoRA with only 0.94% of trainable parameters on some of the tasks in SURE. We further explore the effectiveness of parameter efficient transfer learning for speech synthesis task such as Text-to-Speech (TTS).

![image](https://user-images.githubusercontent.com/35062414/221511052-a6f4c44a-f779-4fca-9142-6ea10254b764.png)

![image](https://user-images.githubusercontent.com/35062414/221511119-27c65410-3086-4509-8927-1ce43efc13af.png)

## Installation
* Set up environments
```python
conda create --name speechprompt python==3.8.5
conda activate speechprompt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
```
* Pip other dependencies
```python
pip install -r requirements.txt
```

## Tasks

* Summary of all tasks and datasets

![image](https://user-images.githubusercontent.com/35062414/221520253-3fba52bf-ff2f-4a2a-8199-be75d4de3989.png)


## How to run
let's specify datasets and arguments now.





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


## Tensorboard
```python
tensorboard --logdir=/data/path/output_earlystop_asr_fleurs_lora_2e3/log --bind_all
```
