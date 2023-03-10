# Dataset
### SNIPS dataset
Here is the download [link](https://huggingface.co/datasets/s3prl/SNIPS)
# Train
Here is an example use SNIPS dataset and fine tuning it.
```python
CUDA_VISIBLE_DEVICES=2,3 python slot_filling.py \
		--dataset snips \
		--data_dir '/data/path/Dataset/SNIPS/' \
		--output_dir '/data/path/output_earlystop_sf_finetune_2e4_scheduler' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--max_steps 50000 \
		--save_steps 5000 \
		--eval_steps 200 \
		--learning_rate 2e-4 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter False \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 8 \
		--gradient_accumulation_steps 1 \
		--per_device_eval_batch_size 8 \
		--num_train_epochs 30 \
		--warmup_ratio 0.1 \
		--logging_steps 100 \
		--logging_dir '/data/path/output_earlystop_sf_finetune_2e4_scheduler/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "slot_type_f1" 
```
We also placed examples according to each training method in "sf.sh", using the following command to start new sf task:
```python
bash sf.sh
```
