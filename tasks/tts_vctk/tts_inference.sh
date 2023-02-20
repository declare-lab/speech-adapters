SAVE_DIR="/data/yingting/libritts/libritts_save_dir_vctk_2e5_finetune_updatefreq1"
# SAVE_DIR="/data/yingting/libritts/pretrained_model/vctk_transformer_phn"
FEATURE_MANIFEST_ROOT="/data/yingting/libritts/feature_manifest_root_vctk"

SPLIT=dev
CHECKPOINT_NAME=best
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
# python average_checkpoints.py --inputs ${SAVE_DIR} \
#   --num-epoch-checkpoints 1 \
#   --output ${CHECKPOINT_PATH}

CUDA_VISIBLE_DEVICES=3 python generate_waveform.py ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --dump-waveforms --dump-target --eval-inference  --results-path "/data/yingting/libritts/results_path/"  #--teacher-forcing  --max-tokens 50000 
