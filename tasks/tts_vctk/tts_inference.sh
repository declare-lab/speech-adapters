SAVE_DIR="/data/yingting/libritts/libritts_save_dir_vctk"
FEATURE_MANIFEST_ROOT="/data/yingting/libritts/feature_manifest_root_vctk"

SPLIT=test
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
python average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

python generate_waveform.py ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --dump-waveforms --results-path "/data/yingting/libritts/results_path/"