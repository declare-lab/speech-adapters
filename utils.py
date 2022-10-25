import librosa
import os
import re
import h5py
import scipy
import numpy as np
from tqdm import tqdm
# import tensorflow as tf
import soundfile as sf
from os.path import join
import random
import pandas as pd
# from tensorflow import keras

# from tensorflow.keras import utils

 

FS = 16000
FFT_SIZE = 512
HOP_LENGTH=256
WIN_LENGTH=512
n_mels = 80

# dir
DATA_DIR = './dataset/ESD/en/'
BIN_DIR = './training_data_en/'



def get_melspectrograms(sound_file, fs=FS, fft_size=FFT_SIZE): 
	# Loading sound file
	y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.
	linear = librosa.stft(y=y,
					 n_fft=fft_size, 
					 hop_length=HOP_LENGTH, 
					 win_length=WIN_LENGTH,
					 window=scipy.signal.hamming,
					 )
	mag = np.abs(linear) #(1+n_fft/2, T)

	# TODO add mel spectrum
	mel_basis = librosa.filters.mel(sr=fs, n_fft=fft_size, n_mels=n_mels)  # (n_mels, 1+n_fft//2)
	mel = np.dot(mel_basis, mag)  # (n_mels, t)
	# shape in (T, 1+n_fft/2)
	return np.transpose(mel.astype(np.float32))  


def read_list(filelist):
	f = open(filelist, 'r')
	Path=[]
	for line in f:
		Path=Path+[line[0:-1]]
	return Path

def read(file_path):
	
	data_file = h5py.File(file_path, 'r')
	mel_sgram = np.array(data_file['mel_sgram'][:])
	
	timestep = mel_sgram.shape[0]
	mel_sgram = np.reshape(mel_sgram,(1, timestep, n_mels))
	
	return {
		'mel_sgram': mel_sgram,
	}   

def pad(array, reference_shape):
	
	result = np.zeros(reference_shape)
	result[:array.shape[0],:array.shape[1],:array.shape[2]] = array

	return result

def filter_file(lists):
	return [f for f in lists if "." not in f]

def get_file_list(dir_root, mode): # train/evaluation/test
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	speaker_dirs = filter_file(os.listdir(dir_root))
	for speaker in speaker_dirs:
		emotion_dirs = filter_file(os.listdir(join(dir_root, speaker)))
		for emotion in emotion_dirs:
			audio_files = os.listdir(join(dir_root, speaker, emotion, mode))
			for audio_file in audio_files:
				audio_dir = join(dir_root, speaker, emotion, mode)
				file_list.append(join(audio_dir, audio_file))
	return file_list

def convert_mp4_2_wav(dir_root):
	one = "/data/yingting/MELD.Raw/output_repeated_splits_test/dia42_utt4.mp4"
	another = "dia42_utt4"
	os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(one, another))
	wav_input, sample_rate = sf.read("dia42_utt4.wav")
	wav_input, sample_rate = librosa.load("dia42_utt4.wav", sr=16000)
	print("wav_input  :", wav_input.shape)
	print("sample_rate:", sample_rate)

	# mp4_train_dir = join(dir_root, "train_splits")
	# mp4_dev_dir = join(dir_root, "dev_splits_complete")
	# mp4_test_dir = join(dir_root, "output_repeated_splits_test")
	# wav_train_dir = join(dir_root, "train")
	# wav_dev_dir = join(dir_root, "dev")
	# wav_test_dir = join(dir_root, "test")
	# if not os.path.exists(wav_train_dir):
	# 	os.makedirs(wav_train_dir)
	# if not os.path.exists(wav_dev_dir):
	# 	os.makedirs(wav_dev_dir)
	# if not os.path.exists(wav_test_dir):
	# 	os.makedirs(wav_test_dir)
	# for mp4_file in os.listdir(mp4_train_dir):
	# 	origin_path = join(mp4_train_dir, mp4_file)
	# 	target_path = join(wav_train_dir, mp4_file[:-4])
	# 	os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(origin_path, target_path))
	# for mp4_file in os.listdir(mp4_dev_dir):
	# 	origin_path = join(mp4_dev_dir, mp4_file)
	# 	target_path = join(wav_dev_dir, mp4_file[:-4])
	# 	os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(origin_path, target_path))
	# for mp4_file in os.listdir(mp4_test_dir):
	# 	origin_path = join(mp4_test_dir, mp4_file)
	# 	target_path = join(wav_test_dir, mp4_file[:-4])
	# 	os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(origin_path, target_path))

def get_meld_files(dir_root, mode):
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	if mode == "train":
		data_dir = join(dir_root, "train")
	elif mode == "evaluation":
		data_dir = join(dir_root, "dev")
	elif mode == "test":
		data_dir = join(dir_root, "test")
	
	for wav_file in os.listdir(data_dir):
		wav_path = join(data_dir, wav_file)
		file_list.append(wav_path)

	return file_list

def get_emo_meld_label(data_dir, mode):  # diaX1_uttX2.wav
	if mode == "train":
		file_name = "train_sent_emo.csv"
	elif mode == "evaluation":
		file_name = "dev_sent_emo.csv"
	elif mode == "test":
		file_name = "test_sent_emo.csv"
	labels = {}
	path = join(data_dir, file_name)
	df_data = pd.read_csv(path) # load the .csv file, specify the appropriate path
	# utt = df_data['Utterance'].tolist() # load the list of utterances
	dia_id = df_data['Dialogue_ID'].tolist() # load the list of dialogue id's
	utt_id = df_data['Utterance_ID'].tolist() # load the list of utterance id's
	emotion = df_data['Emotion'].tolist() # load teh list of emotions
	for i in range(len(emotion)):
		key = 'dia'+str(dia_id[i])+ '_utt' + str(utt_id[i])
		value = emotion[i]
		labels[key] = value
	return labels

def get_iemocap_files(dir_root, mode):  
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	# sessions = [l for l in os.listdir(dir_root) if "Session" in l]
	if mode == "train":
		sessions = ["Session1", "Session2", "Session3"]
	elif mode == "evaluation":
		sessions = ["Session4"]
	else:
		sessions = ["Session5"]
	for session in sessions:
		peoples_dir = join(dir_root, session, "sentences", "wav")
		peoples = [l for l in os.listdir(peoples_dir) if 'Ses' in l]
		peoples = [p for p in peoples if not p.startswith(".")]
		for people in peoples:
			wav_files_dir = join(peoples_dir, people)
			wav_files = [w for w in os.listdir(wav_files_dir) if not w.startswith(".")]
			for wav_file in wav_files:
				wav_path = join(wav_files_dir, wav_file)
				if wav_path.endswith(".wav"):
					file_list.append(wav_path)
	return file_list

def get_iemocap_labels(dir_root):
	info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
	start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []
	# sessions = [l for l in os.listdir(dir_root) if "Session" in l]
	sessions = ["Session1", "Session2", "Session3", "Session4", "Session5"]
	for session in sessions:
		emo_evaluation_dir = join(dir_root, session, "dialog", "EmoEvaluation")
		evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
		for file in evaluation_files:
			with open(join(emo_evaluation_dir, file), 'r', encoding='cp1252') as f:
				content = f.read()
			info_lines = re.findall(info_line, content)
			for line in info_lines[1:]:  # the first line is a header
				start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
				start_time, end_time = start_end_time[1:-1].split('-')
				val, act, dom = val_act_dom[1:-1].split(',')
				val, act, dom = float(val), float(act), float(dom)
				start_time, end_time = float(start_time), float(end_time)
				start_times.append(start_time)
				end_times.append(end_time)
				wav_file_names.append(wav_file_name)
				emotions.append(emotion)
				vals.append(val)
				acts.append(act)
				doms.append(dom)
	return wav_file_names, emotions

def get_vctk_files(dir_root, mode):
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	speakers = filter_file(os.listdir(dir_root))
	for speaker in speakers:
		audio_files = os.listdir(join(dir_root, speaker))
		len_all = len(audio_files)
		len_test = int(len_all * 0.1)
		len_valid = int(len_all * 0.1)
		len_train = len_all - len_test - len_valid
		if mode == "train":
			# audio_files = audio_files[:-100]
			audio_files = audio_files[:len_train]
		elif mode == "evaluation":
			# audio_files = audio_files[-100:-50]
			audio_files = audio_files[len_train:len_train+len_valid]
		else:
			# audio_files = audio_files[-50:]
			audio_files = audio_files[-len_test:]
		for audio_file in audio_files:
			audio_dir = join(dir_root, speaker)
			file_list.append(join(audio_dir, audio_file))

	return file_list



def get_vctk_files_old(dir_root, mode):
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	speaker_dirs = filter_file(os.listdir(dir_root))
	
	test_speaker = ['p360', 'p363', 'p376', 'p351', 'p362', 'p361', 'p374', 'p364']  #this version don't have s5
	train_speaker = list(set(speaker_dirs) - set(test_speaker))

	speakers = None
	if mode == "test":
		speakers = test_speaker
	else:
		speakers = train_speaker

	for speaker in speakers:
		audio_files = os.listdir(join(dir_root, speaker))
		if mode == "train":
			audio_files = audio_files[:-50]
		elif mode == "evaluation":
			audio_files = audio_files[-50:]
		else:
			pass
		for audio_file in audio_files:
			audio_dir = join(dir_root, speaker)
			file_list.append(join(audio_dir, audio_file))

	return file_list


def get_ks_file_list(dir_root, mode):
	if mode not in ['train', 'evaluation', 'test']:
		raise "mode must be in train/evaluation/test!"
	file_list = []
	keywords = filter_file(os.listdir(dir_root))
	# keywords.remove("LICENSE")
	# keywords.remove("_background_noise_")  # need this class or not?
	for keyword in keywords:
		audio_files = os.listdir(join(dir_root, keyword))
		for audio_file in audio_files:
			audio_file_path=join(dir_root, keyword, audio_file)
			file_list.append(audio_file_path)

	all_data_list = file_list
	train_file_lists = []
	test_file_lists = []
	valid_file_lists = []
	testing_kw = set()
	valid_kw = set()
	with open(join(dir_root,'testing_list.txt')) as f:
		test_list = f.readlines()
		for test_f in test_list:
			kw = test_f.split("/")[0]
			testing_kw.add(kw)
			if kw in keywords:
				test_file_lists.append(join(dir_root, test_f.strip('\n')))
	new_test_kw = set()
	for tf in test_file_lists:
		new_test_kw.add(tf.split("/")[-2])
	
	with open(join(dir_root,'validation_list.txt')) as f:
		valid_list = f.readlines()
		for valid_f in valid_list:
			kw = valid_f.split("/")[0]
			valid_kw.add(kw)
			if kw in keywords:
				valid_file_lists.append(join(dir_root, valid_f.strip('\n')))
	new_valid_kw = set()
	for tf in valid_file_lists:
		new_valid_kw.add(tf.split("/")[-2])

	train_file_lists = list((set(all_data_list) - set(test_file_lists)) - set(valid_file_lists))

	if mode == "train":
		return train_file_lists
	elif mode == "evaluation":
		return valid_file_lists
	# elif mode == "test":
	else:
		return test_file_lists


def read_wav(file_lists):
	wav_inputs = []
	labels = []
	emo_label = ['Angry', 'Happy', 'Neutral', 'Surprise', 'Sad']
	for i in range(len(file_lists)):
		wav_input, sample_rate = sf.read(file_lists[i])
		wav_inputs.append(wav_input)
		label = [emo_label.index(str(file_lists[i].split(',')[0].split('/')[-3]))]
		labels.append(label)

	return wav_inputs, labels, sample_rate, emo_label

def read_text(data_dir):
	speaker_dirs = filter_file(os.listdir(data_dir))
	all_texts = {}
	for speaker in speaker_dirs:
		with open(join(data_dir, speaker, f"{speaker}.txt"), 'r', encoding="unicode_escape") as f:
			lines = f.readlines()
			for line in lines:
				line = line.replace('\x00','').strip()
				# print("ascii of line:", ascii(line))
				# print("line		 :", line)
				if line=="":
					pass
				else:
					key, sentence, label = line.strip().split("	")  #.encode('utf8')
					all_texts[key] = sentence
	return all_texts

def read_meld_text(data_dir, mode):
	if mode == "train":
		file_name = "train_sent_emo.csv"
	elif mode == "evaluation":
		file_name = "dev_sent_emo.csv"
	elif mode == "test":
		file_name = "test_sent_emo.csv"
	utterances = {}
	path = join(data_dir, file_name)
	df_data = pd.read_csv(path) # load the .csv file, specify the appropriate path
	utt = df_data['Utterance'].tolist() # load the list of utterances
	dia_id = df_data['Dialogue_ID'].tolist() # load the list of dialogue id's
	utt_id = df_data['Utterance_ID'].tolist() # load the list of utterance id's
	# emotion = df_data['Emotion'].tolist() # load teh list of emotions
	for i in range(len(utt)):
		key = 'dia'+str(dia_id[i])+ '_utt' + str(utt_id[i])
		# value = emotion[i]
		value = utt[i]
		utterances[key] = value
	return utterances


def read_asr_wav(file_lists, texts_dict):
	wav_inputs = []
	texts = []
	missing_key = []
	for i in range(len(file_lists)):
		wav_input, sample_rate = sf.read(file_lists[i])
		
		key = file_lists[i].split("/")[-1][:-4]
		try:
			text = texts_dict[key]
			wav_inputs.append(wav_input)
			texts.append(text)
		except KeyError:
			missing_key.append(key)

	print("len of missing_key:", len(missing_key))

	return wav_inputs, texts, sample_rate

def read_sp_cls_wav(file_lists):
	wav_inputs = []
	labels = []
	speaker_label = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
	for i in range(len(file_lists)):
		wav_input, sample_rate = sf.read(file_lists[i])
		wav_inputs.append(wav_input)
		label = [speaker_label.index(str(file_lists[i].split(',')[0].split('/')[-4]))]
		labels.append(label)

	return wav_inputs, labels, sample_rate, speaker_label

def read_sp_vstk_wav(file_lists):
	wav_inputs = []
	labels = []
	speaker_labels = ['p279', 'p317', 'p266', 'p303', 'p339', 'p254', 'p230', 'p298', \
			'p333', 'p307', 'p274', 'p273', 'p340', 'p295', 'p236', 'p238', 'p241', 'p252', 'p248', \
			'p313', 'p297', 'p287', 'p253', 'p229', 'p233', 'p234', 'p268', 'p306', 'p261', 'p323', \
			'p232', 'p282', 'p286', 'p269', 'p329', 'p250', 'p284', 'p270', 'p237', 'p283', 'p300', \
			'p244', 'p292', 'p345', 'p256', 'p263', 'p304', 'p311', 'p285', 'p262', 'p316', 'p341', \
			'p258', 'p265', 'p305', 'p257', 'p249', 'p259', 'p260', 'p302', 'p245', 'p288', 'p227', \
			'p294', 'p334', 'p264', 'p271', 'p267', 'p330', 'p275', 'p240', 'p281', 'p299', 'p318', \
			'p243', 'p293', 'p277', 'p272', 'p308', 'p276', 'p310', 'p326', 'p239', 'p225', 'p226', \
			'p335', 'p347', 'p343', 'p278', 'p228', 'p231', 'p314', 'p247', 'p312', 'p255', 'p336', \
			'p246', 'p251', 'p301', 'p360', 'p362', 'p374', 'p351', 'p364', 'p361', 'p363', 'p376'] # this version don't have 's5'
	for i in range(len(file_lists)):
		# wav_input, sample_rate = sf.read(file_lists[i])
		wav_input, sample_rate = librosa.load(file_lists[i], sr=16000)
		wav_inputs.append(wav_input)
		label = [speaker_labels.index(str(file_lists[i].split(',')[0].split('/')[-2]))]
		labels.append(label)

	return wav_inputs, labels, sample_rate, speaker_labels

def read_emo_iemocap_wav(file_lists, wav_file_names, emotions, emotion_labels):
	wav_inputs = []
	labels = []
	for i in range(len(file_lists)):
		wav_input, sample_rate = sf.read(file_lists[i])
		wav_file_name = file_lists[i].split(',')[0].split('/')[-1][:-4]
		idx = wav_file_names.index(wav_file_name)
		wav_inputs.append(wav_input)
		labels.append(emotion_labels.index(str(emotions[idx])))
	return  wav_inputs, labels, sample_rate

def read_emo_meld_wav(file_lists, labels_dict, emotion_labels):
	wav_inputs = []
	labels = []
	no_use_count = 0
	for i in range(len(file_lists)):
		# wav_input, sample_rate = sf.read(file_lists[i])
		wav_input, sample_rate = librosa.load(file_lists[i], sr=16000)
		try:
			label = [emotion_labels.index(labels_dict[str(file_lists[i].split(',')[0].split('/')[-1][:-4])])]
			wav_inputs.append(wav_input)
			labels.append(label)
		except:
			no_use_count += 1
	print("no_use_count:", no_use_count)
	return wav_inputs, labels, sample_rate

def read_asr_meld_wav(file_lists, utterances_dict):
	wav_inputs = []
	utterances = []
	missing_key = []
	for i in range(len(file_lists)):
		wav_input, sample_rate = librosa.load(file_lists[i], sr=16000)
		key = str(file_lists[i].split(',')[0].split('/')[-1][:-4])
		try:
			utterance = utterances_dict[key]
			wav_inputs.append(wav_input)
			utterances.append(utterance)
		except KeyError:
			missing_key.append(key)

	print("len of missing_key:", len(missing_key))

	return wav_inputs, utterances, sample_rate
	

def read_ks_cls_wav(file_lists):
	# keyword_label = ['bird', 'four', 'left', 'no', 'forward', 'two', 'cat', 'zero', 'eight', 'five', 'nine', 'marvin', 'stop', 'yes', 'three', 'right', 'dog', 'sheila', 'seven', 'go', 'off', 'bed', 'on', 'learn', 'happy', 'up', 'down', 'six', 'backward', 'one', 'follow', 'wow', 'house', 'tree', 'visual']
	keyword_label = ['off', 'up', 'stop', 'four', 'no', 'down', 'left', 'go', 'yes', 'on', 'right']
	wav_inputs = []
	labels = []
	for i in range(len(file_lists)):
		wav_input, sample_rate = sf.read(file_lists[i])
		wav_inputs.append(wav_input)
		label = [keyword_label.index(str(file_lists[i].split(',')[0].split('/')[-2]))]
		labels.append(label)

	return wav_inputs, labels, sample_rate, keyword_label


def to_categorical(y, num_classes):
	""" 1-hot encodes a tensor """
	return np.eye(num_classes, dtype='uint8')[y]

#train_data = utils.wav_generator(processor, train_list, batch_size=args.per_device_train_batch_size)
def wav_generator(processor, file_list, batch_size=1):
	index=0
	while True:

		filename = [file_list[index+x].split(',')[0] for x in range(batch_size)]

		wav_inputs = []
		for i in range(len(filename)):
			wav_input, sample_rate = sf.read(filename[i]) 
			wav_inputs.append(wav_input)
		
		# pad input values and return pt tensor
		audio_inputs = processor(wav_inputs, padding="longest", sampling_rate=sample_rate, return_tensors="pt")

		emo_class = [emo_label.index(str(file_list[x+index].split(',')[0].split('/')[-3])) for x in range(batch_size)]
		emo_target = to_categorical(emo_class, num_classes=5) # one-hot encoding

		index += batch_size
		if index+batch_size >= len(file_list):
			index = 0
			random.shuffle(file_list)
		
		yield audio_inputs, emo_target

def data_generator(bin_root, file_list, frame=False, batch_size=1):
	index=0
	while True:
			
		filename = [file_list[index+x].split(',')[0] for x in range(batch_size)]
		
		for i in range(len(filename)):
			all_feat = read(join(bin_root,filename[i]+'.h5'))
			sgram = all_feat['mel_sgram']

			# the very first feat
			if i == 0:
				feat = sgram
				max_timestep = feat.shape[1]
			else:
				if sgram.shape[1] > feat.shape[1]:
					# extend all feat in feat
					ref_shape = [feat.shape[0], sgram.shape[1], feat.shape[2]]
					feat = pad(feat, ref_shape)
					feat = np.append(feat, sgram, axis=0)
				elif sgram.shape[1] < feat.shape[1]:
					# extend sgram to feat.shape[1]
					ref_shape = [sgram.shape[0], feat.shape[1], feat.shape[2]]
					sgram = pad(sgram, ref_shape)
					feat = np.append(feat, sgram, axis=0)
				else:
					# same timestep, append all
					feat = np.append(feat, sgram, axis=0)
		
		strength = [float(file_list[x+index].split(',')[1]) for x in range(batch_size)]
		strength=np.asarray(strength).reshape([batch_size])
		frame_strength = np.array([strength[i]*np.ones([feat.shape[1],1]) for i in range(batch_size)])
		# add Multi-task
		emo_class = [emo_label.index(str(file_list[x+index].split(',')[0].split('/')[1])) for x in range(batch_size)]   
		emo_target = to_categorical(emo_class, num_classes=4) # one-hot encoding
		index += batch_size  
		if index+batch_size >= len(file_list):
			index = 0
			random.shuffle(file_list)
		
		if frame:
			yield feat, [strength, frame_strength, emo_target]   
		else:
			yield feat, [strength, emo_target]  
  
			
def extract_to_h5():
	
	print('audio dir: {}'.format(DATA_DIR))
	print('output_dir: {}'.format(BIN_DIR))
		
	if not os.path.exists(BIN_DIR):
		os.makedirs(BIN_DIR)

	count = 0 

	speaker_dirs = os.listdir(DATA_DIR)
	for speaker in speaker_dirs:
		emotion_dirs = os.listdir(join(DATA_DIR, speaker))
		emotion_dirs = [f for f in emotion_dirs if not f.endswith('.txt')]
		for emotion in emotion_dirs:
			split_dirs = os.listdir(join(DATA_DIR, speaker, emotion))
			for split in split_dirs:
				audio_files = os.listdir(join(DATA_DIR, speaker, emotion, split))
				for audio_file in audio_files:
					audio_dir = join(DATA_DIR, speaker, emotion, split)
					out_dir = join(BIN_DIR, speaker, emotion, split)
					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					mel = get_melspectrograms(join(audio_dir, audio_file))
					with h5py.File(join(out_dir, '{}.h5'.format(audio_file)), 'w') as hf:
						hf.create_dataset('mel_sgram', data=mel)
					count += 1
	
	print('start extracting .wav to .h5, {} files found...'.format(count))

			
if __name__ == '__main__':

	# extract_to_h5()
	convert_mp4_2_wav("/data/yingting/MELD.Raw/")