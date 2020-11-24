#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:09:10 2020

@author: root
"""

import pandas as pd
sonyc_path = "../datasets/sonyc-ust/annotations.csv"
annotation_data = pd.read_csv(sonyc_path)

# Get the audio filenames and the splits without duplicates
data = annotation_data[['split', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

train_idxs = []
valid_idxs = []
for idx, (_, row) in enumerate(data.iterrows()):
    if row['split'] == 'train':
        train_idxs.append(idx)
    else:
        valid_idxs.append(idx)
        
        
print("*"*20)
print(f" {len(train_idxs)} train samples")
print("*"*20)

print("*"*20)
print(f" {len(valid_idxs)} test samples")
print("*"*20)