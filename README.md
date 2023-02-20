# Speech_adapters

## Set up
```python
conda create --name speechprompt python==3.8.5
conda activate speechprompt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
pip install huggingface-hub==0.9.1
pip install transformers==4.22.1
pip install datasets==2.4.0
pip install click==8.1.3 
pip install six==1.16.0 
pip install pandas==1.5.0
pip install librosa==0.9.2
pip install h5py==3.7.0
pip install tensorboard==2.10.0
pip install setuptools==59.5.0
pip install adapter-transformers==3.1.0
pip install loralib==0.1.1
pip install jiwer==2.5.1
pip install fairseq==0.12.2
pip install tensorboardX==2.5.1
pip install ipython==8.5.0
pip install path==16.5.0
pip install matplotlib==3.6.1
pip install webrtcvad==2.0.10
pip install editdistance==0.6.0
pip install flashlight==0.1.1
```

## Dataset

### ESD Dataset
download [link](https://github.com/HLTSingapore/Emotional-Speech-Data)

### VCTK Dataset
1. load from huggingface ```load_dataset("vctk", split='train', cache_dir='/data/path/VCTK')```
2. or can download raw data from [link](https://datashare.ed.ac.uk/handle/10283/2651) and follow the data preparation strategy of [nuwave](https://github.com/mindslab-ai/nuwave)
### Google Speech Commands Dataset
download [link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
, also can use this [link](https://github.com/NVIDIA/NeMo/blob/v0.10.1/examples/asr/notebooks/3_Speech_Commands_using_NeMo.ipynb) to download

only use these ['off', 'up', 'stop', 'four', 'no', 'down', 'left', 'go', 'yes', 'on', 'right'] classes, same with [this](https://arxiv.org/ftp/arxiv/papers/2101/2101.04792.pdf) paper 

### VoxCeleb1 Dataset
reference [this](https://github.com/clovaai/voxceleb_trainer)
```python
python ./dataprep.py --save_path /data/yingting/voxceleb1/ --download --user voxceleb1912 --password 0s42xuw6
```


## Train

```python
CUDA_VISIBLE_DEVICES=1 python train.py 
```

```python
CUDA_VISIBLE_DEVICES=1,2,3 python train.py 
```
### tts task
```python
cd tasks/tts_vctk/
bash data_preprocess.sh
bash tts_train.sh
bash tts_inference.sh
```

## Tensorboard
```python
tensorboard --logdir=output/log --bind_all
```
