# CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_finetune_3e5' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--max_steps 50000 \
# 		--save_steps 10000 \
# 		--eval_steps 500 \
# 		--learning_rate 3e-5 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune True \
# 		--per_device_train_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_eval_batch_size 8 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 100 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_finetune_3e5/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_bottleneck_3e4' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--max_steps 50000 \
# 		--save_steps 10000 \
# 		--eval_steps 500 \
# 		--learning_rate 3e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter True \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_eval_batch_size 8 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 100 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_bottleneck_3e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_lora_3e4' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--max_steps 50000 \
# 		--save_steps 10000 \
# 		--eval_steps 500 \
# 		--learning_rate 3e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my False \
# 		--lora_adapter True \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_eval_batch_size 8 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 100 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_lora_3e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_prefixtuning_3e4' \
# 		--group_by_length True \
# 		--do_train True \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--max_steps 50000 \
# 		--save_steps 10000 \
# 		--eval_steps 500 \
# 		--learning_rate 3e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "bottleneck" \
# 		--output_adapter False \
# 		--mh_adapter False \
# 		--prefixtuning False \
# 		--prefix_tuning_my True \
# 		--lora_adapter False \
# 		--feat_enc_adapter False \
# 		--fine_tune False \
# 		--per_device_train_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--per_device_eval_batch_size 8 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_ratio 0.1 \
# 		--logging_steps 100 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_prefixtuning_3e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

CUDA_VISIBLE_DEVICES=0,1 python asr_github.py \
		--dataset "librispeech" \
		--data_dir '/data/yingting/librispeech' \
		--output_dir '/data/yingting/output_earlystop_asr_librispeech_adapterblock_2e4_64' \
		--group_by_length True \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--fp16 True \
		--gradient_checkpointing True \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--max_steps 50000 \
		--save_steps 10000 \
		--eval_steps 500 \
		--learning_rate 2e-4 \
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
		--gradient_accumulation_steps 1 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 30 \
		--weight_decay=0.005 \
		--warmup_ratio 0.1 \
		--logging_steps 100 \
		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_adapterblock_2e4_64/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "wer" \
		--greater_is_better False

# CUDA_VISIBLE_DEVICES=0,1 python asr_new.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_bottleneck_2e5' \
# 		--group_by_length True \
# 		--do_train False \
# 		--do_eval True \
# 		--do_predict False \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 200 \
# 		--eval_steps 100 \
# 		--learning_rate 2e-4 \
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
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_bottleneck_2e5/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=2,3 python asr_new.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_prefix_2e3' \
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
# 		--per_device_train_batch_size 64 \
# 		--gradient_accumulation_steps 4 \
# 		--per_device_eval_batch_size 64 \
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_prefix_2e3/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=1,3 python asr_new.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/output_earlystop_asr_librispeech_tiny_att_2e4' \
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
# 		--learning_rate 2e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "conv_adapter" \
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
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/output_earlystop_asr_librispeech_tiny_att_2e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False

# CUDA_VISIBLE_DEVICES=2,3 python asr_new.py \
# 		--dataset "librispeech" \
# 		--data_dir '/data/yingting/librispeech' \
# 		--output_dir '/data/yingting/Output/output_earlystop_asr_librispeech_newadapterblock_2e4' \
# 		--group_by_length True \
# 		--do_train False \
# 		--do_eval False \
# 		--do_predict True \
# 		--fp16 True \
# 		--gradient_checkpointing True \
# 		--evaluation_strategy "steps" \
# 		--save_strategy "steps" \
# 		--save_steps 200 \
# 		--eval_steps 100 \
# 		--learning_rate 2e-4 \
# 		--feat_adapter_name "conv_adapter" \
# 		--trans_adapter_name "adapterblock" \
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
# 		--num_train_epochs 30 \
# 		--weight_decay=0.005 \
# 		--warmup_steps=1000 \
# 		--logging_steps 20 \
# 		--logging_dir '/data/yingting/Output/output_earlystop_asr_librispeech_newadapterblock_2e4/log' \
# 		--load_best_model_at_end True \
# 		--metric_for_best_model "wer" \
# 		--greater_is_better False