import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml
from collections import OrderedDict

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


from urban_sound_tagging_baseline.classify import get_file_targets, get_subset_split, generate_output_file
from urban_sound_tagging_baseline.metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

from scipy.stats import describe
from sklearn.ensemble import RandomForestClassifier


class MyDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __getitem__(self, index):
		x = torch.Tensor(self.x[index])
		y = torch.Tensor(self.y[index])
		return (x, y)

	def __len__(self):
		count = self.x.shape[0]
		return count

def fit_mel_size(mel, mel_frames=998):
	if mel.shape[0]<mel_frames:
		padding_len = mel_frames-mel.shape[0]
		zero_pad = np.zeros((padding_len, mel.shape[1]))
		mel = np.vstack((mel, zero_pad))
	elif mel.shape[0]>mel_frames:
		mel = mel[:mel_frames,:]
	
	return mel


def load_mels(file_list, mel_dir):
	
	mel_list = []
	
	for idx, filename in enumerate(file_list):
		mel_path = os.path.join(mel_dir, os.path.splitext(filename)[0] + '.npy')
		mel = fit_mel_size(np.load(mel_path), mel_frames = 998)
		mel_list.append(mel)

	return mel_list

def prepare_data(train_file_idxs, test_file_idxs, mel_list,
						   target_list):
	
	"""
	modified prepare_framewise_data() in classify.py of the baseline code
	"""
	X_train = []
	y_train = []
	for idx in train_file_idxs:

		X_train.append(mel_list[idx])
		y_train.append(target_list[idx])

	train_idxs = np.random.permutation(len(X_train))

	X_train = np.array(X_train)[train_idxs]
	y_train = np.array(y_train)[train_idxs]

	X_valid = []
	y_valid = []
	for idx in test_file_idxs:

		X_valid.append(mel_list[idx])

		y_valid.append(target_list[idx])

	test_idxs = np.random.permutation(len(X_valid))
	X_valid = np.array(X_valid)[test_idxs]
	y_valid = np.array(y_valid)[test_idxs]

	return X_train, y_train, X_valid, y_valid

def predict(mel_list, test_file_idxs, clf):
	"""
	Modified predict_framewise() in classify.py of the baseline code

	"""
	test_x = np.array(mel_list)[np.array(test_file_idxs)]
	test_x = np.reshape(test_x,(-1,1,test_x.shape[1],test_x.shape[2]))
	dtest_x = np.diff(test_x, n=1, axis=-2)
	d2test_x = np.diff(test_x, n=2, axis=-2)
	n_samples, (minim, maxim), mean, var, skewness, kurtosis = describe(test_x, axis=-2, ddof=1,
																		bias=True, nan_policy='propagate')
	median = np.median(test_x, axis=2)
	mean_d = np.mean(dtest_x, axis=2)
	var_d = np.var(dtest_x, axis=2)
	mean_d2 = np.mean(d2test_x, axis=2)
	var_d2= np.var(d2test_x, axis=2)
	test_feat = np.concatenate([minim, maxim, mean, var, skewness,
						  kurtosis, median, mean_d, var_d, mean_d2, var_d2], axis=1)
	print(test_feat.shape)
	test_feat = test_feat.reshape(test_feat.shape[0], -1)
	print(test_feat.shape)
	model_output = clf.predict(test_feat)

	y_pred = [i for i in model_output]

	return y_pred


def load_pretrained_weights(current_model, pretrained_model_path):
	pretrained_state_dict = torch.load(pretrained_model_path)
	new_state_dicts = OrderedDict()
	model_state_dict = current_model.state_dict()

	for k, v in pretrained_state_dict.items():
		if k in model_state_dict.keys():
			new_state_dicts[k] = v

	model_state_dict.update(new_state_dicts)
	current_model.load_state_dict(model_state_dict)

	return current_model

def freeze_layaer(layer):
	for param in layer.parameters():
		param.requires_grad = False

def train(annotation_path, taxonomy_path, mel_dir, models_dir, output_dir,
		batch_size, num_epochs, learning_rate, patience):
	"""
	This function is based on train_framewise() in the baseline code.

	"""



	os.makedirs(models_dir, exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)


	# Load annotations and taxonomy
	print("* Loading dataset.")
	annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
	with open(taxonomy_path, 'r') as f:
		taxonomy = yaml.load(f, Loader=yaml.Loader)

	file_list = annotation_data['audio_filename'].unique().tolist()

	coarse_target_labels = []
	for k0, fine_dict in taxonomy['fine'].items():
		for k1,v1 in fine_dict.items():
			key = "-".join([str(k0),str(k1)])
			coarse_target_labels.append("_".join([key, v1]))

	coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
	train_file_idxs, test_file_idxs = get_subset_split(annotation_data)

	target_list = coarse_target_list
	labels = coarse_target_labels
	n_classes = len(coarse_target_labels)

	print('load mel spectrograms')
	mel_list = load_mels(file_list, mel_dir)

	print('prepare data')

	train_X, train_y, val_X, val_y = prepare_data(train_file_idxs, test_file_idxs, mel_list,target_list)

	train_y = train_y.astype('int32')
	val_y = val_y.astype('int32')

	print(train_X.shape)

	#(num of examples, channel, frames, frequency bands)
	train_X = np.reshape(train_X,(-1,1,train_X.shape[1],train_X.shape[2]))
	val_X = np.reshape(val_X,(-1,1,val_X.shape[1],val_X.shape[2]))

	#######################################################
	dtrain_X = np.diff(train_X, n=1, axis=2)
	d2train_X = np.diff(train_X, n=2, axis=2)
	n_samples, (minim, maxim), mean, var, skewness, kurtosis = describe(train_X, axis=2, ddof=1,																		bias=True, nan_policy='propagate')
	median = np.median(train_X, axis=2)
	mean_d = np.mean(dtrain_X, axis=2)
	var_d = np.var(dtrain_X, axis=2)
	mean_d2 = np.mean(d2train_X, axis=2)
	var_d2= np.var(d2train_X, axis=2)
	
	train_feat = np.concatenate([minim, maxim, mean, var, skewness,
						   kurtosis, median, mean_d, var_d, mean_d2, var_d2], axis=1)
	print(train_feat.shape)
	train_feat = train_feat.reshape(train_feat.shape[0], -1)
	print(train_feat.shape)
	clf = RandomForestClassifier(n_estimators=500)
	clf.fit(train_feat, train_y)
	
	y_pred = predict(mel_list, test_file_idxs, clf)

	aggregation_type = 'max'
	label_mode = 'coarse'
	generate_output_file(y_pred, test_file_idxs, output_dir, file_list,
						 aggregation_type, label_mode, taxonomy)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path")
	parser.add_argument("taxonomy_path")
	parser.add_argument("mel_dir", type=str)
	parser.add_argument("models_dir", type=str)
	parser.add_argument("output_dir", type=str)

	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_epochs", type=int, default=100)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--patience", type=int, default=10)


	args = parser.parse_args()


	train(args.annotation_path,
		args.taxonomy_path,
		args.mel_dir,
		args.models_dir,
		args.output_dir,
		batch_size=args.batch_size,
		num_epochs=args.num_epochs,
		learning_rate=args.learning_rate,
		patience=args.patience)