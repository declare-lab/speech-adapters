# CUDA_VISIBLE_DEVICES=3,2 python train.py \
# 		--dataset "meld" \
# 		--data_dir "/data/yingting/MELD.Raw" \
# 		--output_dir '/data/yingting/output_earlystop_emo_cls_meld_finetune_7e6' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 7e-6 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
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
# 		--logging_dir '/data/yingting/output_earlystop_emo_cls_meld_finetune_7e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=3,2 python train.py \
# 		--dataset "meld" \
# 		--data_dir "/data/yingting/MELD.Raw" \
# 		--output_dir '/data/yingting/output_earlystop_emo_cls_meld_bottleneck_8e6' \
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
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefixtuning False \
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
# 		--logging_dir '/data/yingting/output_earlystop_emo_cls_meld_bottleneck_8e6/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=3,2 python train.py \
# 		--dataset "meld" \
# 		--data_dir "/data/yingting/MELD.Raw" \
# 		--output_dir '/data/yingting/output_earlystop_emo_cls_meld_prefixtuningmy_2e2' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 2e-2 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
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
# 		--logging_dir '/data/yingting/output_earlystop_emo_cls_meld_prefixtuningmy_2e2/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

# CUDA_VISIBLE_DEVICES=3,2 python train.py \
# 		--dataset "meld" \
# 		--data_dir "/data/yingting/MELD.Raw" \
# 		--output_dir '/data/yingting/output_earlystop_emo_cls_meld_lora_8e4' \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 500 \
# 		--eval_steps 25 \
# 		--learning_rate 8e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
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
# 		--logging_dir '/data/yingting/output_earlystop_emo_cls_meld_lora_8e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "accuracy" 

CUDA_VISIBLE_DEVICES=2,3 python train.py \
		--dataset "meld" \
		--data_dir "/data/yingting/MELD.Raw" \
		--output_dir '/data/yingting/output_earlystop_ser_meld_adapterblock_2e3' \
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
		--prefixtuning False \
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
		--logging_dir '/data/yingting/output_earlystop_ser_meld_adapterblock_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "f1" 
