
for lr in 2e-7 #2e-2 2e-3 2e-4 2e-5 2e-6 2e-7
do
	CUDA_VISIBLE_DEVICES=2,3 python train.py \
		--output_dir '/data/yingting/output_earlystop_emo_cls_adapterblock_'${lr}'' \
		--do_train False \
		--do_eval False \
		--do_predict True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate $lr \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "adapterblock" \
		--output_adapter True \
		--mh_adapter False \
		--prefixtuning False \
		--prefix_tuning_my False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 5 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_emo_cls_adapterblock_'${lr}'/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 
done