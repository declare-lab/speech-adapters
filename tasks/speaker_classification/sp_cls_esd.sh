
# CUDA_VISIBLE_DEVICES=0,1 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/Dataset/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_finetune_8e6' \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 8e-6 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune True \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_finetune_8e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=0,1 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/Dataset/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_bottleneck_8e6' \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 8e-6 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_bottleneck_8e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=0,1 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_lora_8e6' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 8e-6 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefix_tuning_my False \
# 		--lora_adapter True \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_lora_8e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=0,1 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_prefixtuningmy_8e6' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 8e-6 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefix_tuning_my True \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_prefixtuningmy_8e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/Dataset/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_adapterblock_2e4' \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 2e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "adapterblock" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefix_tuning_my True \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_adapterblock_2e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy"

# CUDA_VISIBLE_DEVICES=0,1 python speaker_recg.py \
# 		--dataset esd\
# 		--data_dir "/data/yingting/ESD/en/" \
# 		--output_dir '/data/yingting/output_earlystop_sp_esd_convadapter_bottleneck_2e4' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 2e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter True \
# 		--fine_tune False \
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 100 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_sp_esd_convadapter_bottleneck_2e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy"


CUDA_VISIBLE_DEVICES=0,3 python speaker_recg.py \
		--dataset esd \
		--data_dir "/data/yingting/Dataset/ESD/en/" \
		--output_dir '/data/yingting/output_earlystop_sr_esd_adapterblock_2e3_scale16' \
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
		--output_adapter True \
		--mh_adapter False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sr_esd_adapterblock_2e3_scale16/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy"


