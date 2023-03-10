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
First, we need to specify datasets and arguments. let's use "esd" as the dataset, "finetune" as the tuning method and "speech emotion recognition" as an example:

```python
CUDA_VISIBLE_DEVICES=2,3 python train.py \
		--dataset "esd" \
		--data_dir "/data/path/ESD" \
		--output_dir '/data/path/output_earlystop_ser_esd_finetune_2e3' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/path/output_earlystop_ser_esd_finetune_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "f1" 
```

#### Parameters

* dataset: specify the dataset, such as "esd", "fleurs", "fluent_commands", etc.
* data_dir: path to the dataset file, for instance, "../data/path/ESD" 
* output_dir: path to the checkpoints and logs, for instance, '../data/path/output_earlystop_ser_esd_finetune_2e3' 
* do_train: True if want to train 
* do_eval: True if want to eval 
* do_predict: True if want to inference
* evaluation_strategy: It can be set according to the official setting of huggingface
* save_strategy: It can be set according to the official setting of huggingface 
* save_steps: It can be set according to the official setting of huggingface
* eval_steps: It can be set according to the official setting of huggingface
* learning_rate: It can be set according to the official setting of huggingface
* feat_adapter_name: The adapter type added in the features encoder, but not applied to this article, can be skipped 
* trans_adapter_name: The adapter type added in transformer layer, such as "adapterblock" for ConvAdapter and "bottleneck" for Bottleneck Adapter
* output_adapter: True if added after feedforward of every transformer layer, only control ConvAdapter and Bottleneck Adapter
* mh_adapter: True if added after multihead attention of every transformer layer, only control ConvAdapter and Bottleneck Adapter 
* prefix_tuning: True if prefix-tuning is added 
* lora_adapter: True if Lora is added 
* feat_enc_adapter: True if adapter is add in features encoder of wav2vec2 
* fine_tune: True if only need fine tuning 
* per_device_train_batch_size: It can be set according to the official setting of huggingface  
* gradient_accumulation_steps: It can be set according to the official setting of huggingface  
* per_device_eval_batch_size: It can be set according to the official setting of huggingface 
* num_train_epochs: It can be set according to the official setting of huggingface  
* warmup_ratio: It can be set according to the official setting of huggingface  
* logging_steps: It can be set according to the official setting of huggingface  
* logging_dir: It can be set according to the official setting of huggingface 
* load_best_model_at_end: It can be set according to the official setting of huggingface 
* metric_for_best_model: It can be set according to the official setting of huggingface 

#### Emotion classification
Let's further explain the five training methods of the model.For example, start a new emotion classification task, we will set the corresponding parameter like below:
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
## ConvAdapter
--trans_adapter_name "adapterblock"
--output_adapter True
```

We also placed examples according to each training method in "emotion_cls.sh", using the following command to start new emotion classification task:
```python
bash emotion_cls.sh
```

## Tensorboard
In order to further supervise the convergence of model training, we can view the log file through Tensorboard:
```python
tensorboard --logdir=/data/path/output_earlystop_asr_fleurs_lora_2e3/log --bind_all
```
