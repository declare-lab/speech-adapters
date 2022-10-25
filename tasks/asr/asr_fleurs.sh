CUDA_VISIBLE_DEVICES=2,3 python asr_new.py \
		--dataset "fleurs" \
		--data_dir '/data/yingting/fleurs' \
		--output_dir '/data/yingting/output_earlystop_asr_fleurs_finetune_2e6' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 2e-6 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 4 \
		--gradient_accumulation_steps 8 \
		--per_device_eval_batch_size 4 \
		--num_train_epochs 20 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_asr_fleurs_finetune_2e6/log' \
		# --load_best_model_at_end True \
		# --metric_for_best_model "wer" \
		# --greater_is_better False