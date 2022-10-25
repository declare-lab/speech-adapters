AUDIO_DATA_ROOT="/data/yingting/libritts/LibriTTS/train-clean-100/103/1241"
AUDIO_MANIFEST_ROOT="/data/yingting/libritts/audio_manifest_root_vctk"
PROCESSED_DATA_ROOT="/data/yingting/libritts/processed_data_root_vctk"
SAVE_DIR="/data/yingting/libritts/libritts_save_dir_vctk"

# """ 1. Download data, create splits and generate audio manifests"""

# python get_libritts_audio_manifest.py \
# 	--output-data-root ${AUDIO_DATA_ROOT} \
# 	--output-manifest-root ${AUDIO_MANIFEST_ROOT}

# """ 2. To denoise audio and trim leading/trailing silence using signal processing based VAD, which generates 
# a new audio TSV manifest under ${PROCESSED_DATA_ROOT} with updated path to the processed audio and a new column for SNR."""

# for SPLIT in dev test train; do
#     python denoise_and_vad_audio.py \
# 		--audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
# 		--output-dir ${PROCESSED_DATA_ROOT} \
# 		--denoise --vad --vad-agg-level 3
# done

CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
EVAL_OUTPUT_ROOT="/data/yingting/libritts/eval_output_root_vctk"
WAV2VEC2_CHECKPOINT_PATH="/data/yingting/libritts/wav2vec2_checkpoint/wav2vec_small_10m.pt"
WAV2VEC2_DICT_DIR="/data/yingting/libritts/wav2vec2_checkpoint/"

# """ 3. do filtering by CER, follow the Automatic Evaluation section to run ASR model """

# for SPLIT in dev test train; do
# 	python get_eval_manifest.py \
# 		--generation-root ${SAVE_DIR}/generate-${CHECKPOINT_NAME}-${SPLIT} \
# 		--audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
# 		--output-path ${EVAL_OUTPUT_ROOT}/${SPLIT}_16khz.tsv \
# 		--vocoder griffin_lim --sample-rate 16000 --audio-format flac \
# 		--use-resynthesized-target \
# 		--eval-target

# 	python eval_asr.py \
# 		--audio-header syn --text-header text --err-unit char --split ${SPLIT} \
# 		--w2v-ckpt ${WAV2VEC2_CHECKPOINT_PATH} --w2v-dict-dir ${WAV2VEC2_DICT_DIR} \
# 		--raw-manifest ${EVAL_OUTPUT_ROOT}/${SPLIT}_16khz.tsv --asr-dir ${EVAL_OUTPUT_ROOT}/asr
# done

# """ 4. concat all splits' cer file"""

# python cat_splits_cer.py --train-cer-tsv ${EVAL_OUTPUT_ROOT}/asr/uer_char.train.tsv \
# 	--dev-cer-tsv ${EVAL_OUTPUT_ROOT}/asr/uer_char.dev.tsv \
# 	--test-cer-tsv ${EVAL_OUTPUT_ROOT}/asr/uer_char.test.tsv \
# 	--uer-char-tsv ${EVAL_OUTPUT_ROOT}/asr/uer_char.tsv

# """ 5. extract log-Mel spectrograms, generate feature manifest and create data configuration YAML """

FEATURE_MANIFEST_ROOT="/data/yingting/libritts/feature_manifest_root_vctk"

python get_feature_manifest.py \
	--audio-manifest-root ${PROCESSED_DATA_ROOT} \
	--output-root ${FEATURE_MANIFEST_ROOT} \
	--use-g2p \
	--snr-threshold 15 \
	--cer-threshold 0.1 --cer-tsv-path ${EVAL_OUTPUT_ROOT}/asr/uer_char.tsv  ### 1.modify 'uer_cer' to 'uer_char', add '/asr', should be right 2. --ipa-vocab