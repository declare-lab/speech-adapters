for lr in 2e-1 #2e-3 2e-4 2e-5 2e-6 2e-7
do
	CUDA_VISIBLE_DEVICES=2,3 python asr.py \
		--output_dir '/data/yingting/output_earlystop_asr_prefixtuningmy_'${lr}'' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate $lr \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning_my True \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 5 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_asr_prefixtuningmy_'${lr}'/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False
done