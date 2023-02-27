##### Fine-tune ######
CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/path/output_earlystop_asr_fleurs_finetune_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/path/Dataset/fleurs' \
		--group_by_length True \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/path/output_earlystop_asr_fleurs_finetune_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \

##### Bottleneck ######
CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/path/output_earlystop_asr_fleurs_bottleneck_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/path/Dataset/fleurs' \
		--group_by_length True \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter True \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/path/output_earlystop_asr_fleurs_bottleneck_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \

##### Prefix-tuning ######
CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/path/output_earlystop_asr_fleurs_prefixtuning_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/path/Dataset/fleurs' \
		--group_by_length True \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning True \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/path/output_earlystop_asr_fleurs_prefixtuning_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \

##### Lora ######
CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/path/output_earlystop_asr_fleurs_lora_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/path/Dataset/fleurs' \
		--group_by_length True \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter True \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/path/output_earlystop_asr_fleurs_lora_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \

##### Adapterblock ######
CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/path/output_earlystop_asr_fleurs_adapterblock_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/path/Dataset/fleurs' \
		--group_by_length True \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter True \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/path/output_earlystop_asr_fleurs_adapterblock_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \