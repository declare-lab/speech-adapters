# Dataset

### Google Speech Commands Dataset
Here is the download [link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
, also can use this [link](https://github.com/NVIDIA/NeMo/blob/v0.10.1/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb) to download.

Only use these ['off', 'up', 'stop', 'four', 'no', 'down', 'left', 'go', 'yes', 'on', 'right'] classes, same as in [this](https://arxiv.org/ftp/arxiv/papers/2101/2101.04792.pdf) paper
# Train
Here is an example use speech commands dataset and fine tuning it.
```python
CUDA_VISIBLE_DEVICES=0,1 python keyword_spotting.py \
		--output_dir '/data/path/output_earlystop_ks_finetune_8e6' \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 8e-6 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
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
		--logging_dir '/data/path/output_earlystop_ks_finetune_8e6/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 
```
We also placed examples according to each training method in "ks.sh", using the following command to start new ks task:
```python
bash ks.sh
```
