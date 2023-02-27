
##### Fine-tune ######
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sp_finetune_8e6' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 8e-6 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune True \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sp_finetune_8e6/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 

##### Bottleneck ######
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sp_bottleneck_8e6' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 8e-6 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter True \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sp_bottleneck_8e6/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 

##### Lora ######
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sp_lora_2e4' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 2e-4 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning False \
		--lora_adapter True \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sp_lora_2e4/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 

##### Prefix-tuning ######
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sp_prefix_2e3' \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--evaluation_strategy "steps" \
		--save_strategy "steps" \
		--save_steps 500 \
		--eval_steps 25 \
		--learning_rate 2e-3 \
		--feat_adapter_name "conv_adapter" \
		--trans_adapter_name "bottleneck" \
		--output_adapter False \
		--mh_adapter False \
		--prefix_tuning True \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sp_prefix_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 

##### Adapterblock ######
CUDA_VISIBLE_DEVICES=2,3 python speaker_recg.py \
		--dataset vctk\
		--data_dir "/data/yingting/VCTK_Wav/wav48/" \
		--output_dir '/data/yingting/output_earlystop_sr_vctk_adapterblock_2e3' \
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
		--prefix_tuning False \
		--lora_adapter False \
		--feat_enc_adapter False \
		--fine_tune False \
		--per_device_train_batch_size 64 \
		--gradient_accumulation_steps 4 \
		--per_device_eval_batch_size 64 \
		--num_train_epochs 100 \
		--warmup_ratio 0.1 \
		--logging_steps 20 \
		--logging_dir '/data/yingting/output_earlystop_sr_vctk_adapterblock_2e3/log' \
		--load_best_model_at_end True \
		--metric_for_best_model "accuracy" 


