# Dataset
### ESD Dataset
Here is the download [link](https://github.com/HLTSingapore/Emotional-Speech-Data)
### VCTK Dataset
1. load from huggingface ```load_dataset("vctk", split='train', cache_dir='/data/path/VCTK')```
2. or can download raw data from [link](https://datashare.ed.ac.uk/handle/10283/2651) and follow the data preparation strategy of [nuwave](https://github.com/mindslab-ai/nuwave)
# Train
Here is an example use vctk dataset and fine tuning it.
```python
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sp_finetune_8e6' \
		--do_train True \
		--do_eval True \
		--do_predict False \
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
		--logging_dir '/data/yingting/output_earlystop_sp_finetune_8e6/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 
```
We also placed examples according to each training method in "sp_cls.sh", using the following command to start new sr task:
```python
bash sp_cls.sh
```
