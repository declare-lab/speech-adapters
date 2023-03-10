# TTS
We use [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS), choice "transformer" as the backbone model, and implement fine-tune, prefix-tuning, lora, bottleneck adapter, and convadapter on it.

One can follow the README in Comprehensive-Transformer-TTS, and place the "LTS"(for LibriTTS) and "L2ARCTIC"(for L2ARCTIC) under "config" folder, and replace the file "../model/transformer.py" in library use "transformer.py"。

We use the checkpoint trained by 900000 steps on VCTK dataset, and finetune LTS and L2ARCTIC both 4000 steps. 
