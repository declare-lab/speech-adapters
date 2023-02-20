# CUDA_VISIBLE_DEVICES=0,1 python phoneme_recognition.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/hf_datasets' \
# 		--output_dir '/data/yingting/Output/output_earlystop_pr_librispeech_lora_2e2' \
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
# 		--learning_rate 2e-2 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter True \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 16 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 16 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/Output/output_earlystop_pr_librispeech_lora_2e2/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "per" \
# 		--greater_is_better False

CUDA_VISIBLE_DEVICES=0,3 python phoneme_recognition.py \
		--dataset "librispeech" \
		--data_dir '/data/yingting/hf_datasets' \
		--output_dir '/data/yingting/Output/output_earlystop_pr_librispeech_bottleneck_2e3' \
		--group_by_length True \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 200 \
		--eval_steps 100 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter True \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 16 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 16 \
		--num_train_epochs 100 \
		--weight_decay=0.005 \
		--warmup_steps=1000 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/Output/output_earlystop_pr_librispeech_bottleneck_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "per" \
		--greater_is_better False