# CUDA_VISIBLE_DEVICES=0,1 python asr_new.py \
# 		--output_dir '/data/yingting/output_earlystop_asr_fleurs_finetune_1e4' \
# 		--dataset "fleurs" \
# 		--data_dir '/data/yingting/fleurs' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 100 \
# 		--eval_steps 100 \
# 		--learning_rate 1e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune True \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 50 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_finetune_1e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False \
# 		# --gradient_accumulation_steps 4 \

# CUDA_VISIBLE_DEVICES=0,1 python asr_new.py \
# 		--output_dir '/data/yingting/output_earlystop_asr_fleurs_prefix_2e3' \
# 		--dataset "fleurs" \
# 		--data_dir '/data/yingting/fleurs' \
# 		--group_by_length True \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--fp16 True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 100 \
# 		--eval_steps 100 \
# 		--learning_rate 2e-3 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my True \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 50 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_prefix_2e3/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False \
# 		--gradient_checkpointing True \
# 		# --gradient_accumulation_steps 4 \


# CUDA_VISIBLE_DEVICES=0,1 python asr_new.py \
# 		--output_dir '/data/yingting/output_earlystop_asr_fleurs_adapterblock_6e3' \
# 		--dataset "fleurs" \
# 		--data_dir '/data/yingting/fleurs' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 200 \
# 		--eval_steps 100 \
# 		--learning_rate 6e-3 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "adapterblock" \
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 50 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_adapterblock_6e3/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False \
# 		# --gradient_accumulation_steps 4 \



# CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
# 		--output_dir '/data/yingting/output_earlystop_asr_fleurs_bottleneck_1e4' \
# 		--dataset "fleurs" \
# 		--data_dir '/data/yingting/Dataset/fleurs' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict True \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 100 \
# 		--learning_rate 1e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--num_train_epochs 60 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 50 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_bottleneck_1e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False 
# 		# --gradient_accumulation_steps 4 \

CUDA_VISIBLE_DEVICES=2,3 python asr_github.py \
		--output_dir '/data/yingting/output_earlystop_asr_fleurs_lora_2e3' \
		--dataset "fleurs" \
		--data_dir '/data/yingting/Dataset/fleurs' \
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
		--prefixtuning False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--num_train_epochs 50 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 50 \
		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_lora_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False 
		# --gradient_accumulation_steps 4 \