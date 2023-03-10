# Dataset
## Fluent Speech Commands 
Here is the download [link](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
# Train
Here is an example use fluent speech commands dataset and fine tuning it.
```python
CUDA_VISIBLE_DEVICES=0,1 python intent_cls.py \
		--dataset fluent_commands \
		--data_dir '/data/path/Dataset/fluent_speech_commands_dataset' \
		--output_dir '/data/path/Output/output_earlystop_ic_finetune_2e4' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 2e-4 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 8 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 8 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/path/Output/output_earlystop_ic_finetune_2e4/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "acc"
```
	
We also placed examples according to each training method in "ic.sh", using the following command to start new asr task:
```python
bash ic.sh
```
