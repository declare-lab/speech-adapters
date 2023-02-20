CUDA_VISIBLE_DEVICES=0,2 python asr_new.py \
		--dataset "voxpopuli" \
		--data_dir '/data/yingting/voxpopuli' \
		--output_dir '/data/yingting/output_earlystop_asr_voxpopuli_finetune_2e5' \
		--group_by_length True \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 100 \
		--eval_steps 25 \
		--learning_rate 2e-5 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 2 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_asr_voxpopuli_finetune_2e5/log' \
		# --load_best_model_at_end True \
		# --metric_for_best_model "wer" \
		# --greater_is_better False