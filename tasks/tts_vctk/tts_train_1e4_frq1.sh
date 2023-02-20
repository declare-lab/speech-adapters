FEATURE_MANIFEST_ROOT="/data/yingting/libritts/feature_manifest_root_vctk"
SAVE_DIR="/data/yingting/libritts/libritts_save_dir_vctk_1e4_finetune_updatefreq1"

rm -rf /data/yingting/libritts/libritts_save_dir_vctk_1e4_finetune_updatefreq1/*

CUDA_VISIBLE_DEVICES=3 python tts.py ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4  --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 1 --eval-inference --best-checkpoint-metric mcd_loss \
  --skip-invalid-size-inputs-valid-test --batch-size 64 --reset-dataloader \
  --batch-size-valid 64 --validate-interval 10 --save-interval 10 #--max-tokens 30000