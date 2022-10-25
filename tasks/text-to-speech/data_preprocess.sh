AUDIO_DATA_ROOT="/data/yingting/libritts/LibriTTS/train-clean-100/103/1241"
AUDIO_MANIFEST_ROOT="/data/yingting/libritts/audio_manifest_root_ljspeech"
FEATURE_MANIFEST_ROOT="/data/yingting/libritts/feature_manifest_root_ljspeech"

python get_libritts_audio_manifest.py \
	--output-data-root ${AUDIO_DATA_ROOT} \
	--output-manifest-root ${AUDIO_MANIFEST_ROOT}

python get_libritts_feature_manifest.py \
	--audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
	--output-root ${FEATURE_MANIFEST_ROOT} \
	--ipa-vocab --use-g2p