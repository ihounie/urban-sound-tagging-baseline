#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:18:46 2020

@author: root
"""
from pathlib import Path
import pandas as pd
import wget
import os
from process_jsons import *
import shutil
import sys

rootdir_path = '../'
mavd_path = "MAVD2"
sonyc_path = "/home/hounie/audio/DCASE-models/datasets/SONYC_UST"
sonyc_audio_path = os.path.join(sonyc_path, "audio")
output_path = "/home/hounie/audio/urban-sound-tagging-baseline/sonyc-mavd/data"
import sys
sys.path.append(rootdir_path)
from dcase_models.data.datasets import SONYC_UST
dataset = SONYC_UST(sonyc_path)
dataset.download()  

extract=False
copy=True

sys.path.append(rootdir_path)

def copy_all_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)          
if extract:
    process_mavd(duration = 10, #seconds,
                 clip_length = 10,
                 min_label = 3,
                 overlap  = 1,
                mavd_path = mavd_path, split="train", train_frac = 0.5, tagless_frac = 2 )
    jsons2csv(mode="fine",mavd_path = mavd_path, split="train", split_out ="validate")

annot_csv_pth_train = os.path.join(mavd_path, "validate","annotations.csv")
mavd = pd.read_csv(annot_csv_pth_train)#, index_col=[0])

sonyc = pd.read_csv(os.path.join(sonyc_path, 'annotations.csv'))#, index_col=[0])
print(sonyc.head())
sonyc = sonyc[sonyc["split"] == "train"]
print(sonyc.head())
print(f" samples from SONYC: {len(sonyc.index)}")
print(f" samples from MAVD: {len(mavd.index)}")
print(set(sonyc.columns) - set(mavd.columns))
print(set(mavd.columns) - set(sonyc.columns))

merged = pd.concat([mavd, sonyc])
print(f"Total samlples: {len(merged.index)}")
print(f" rows lost (should be zero): {len(merged.index)-len(mavd.index)-len(sonyc.index)}")

Path(output_path).mkdir(parents=True, exist_ok=True)
print(os.getcwd())
merged.to_csv(os.path.join(output_path, 'annotations.csv'), index=False)

out_train_dir = os.path.join(output_path,"train")
out_val_dir = os.path.join(output_path,"validate")
pathlib.Path(out_train_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(out_val_dir).mkdir(parents=True, exist_ok=True)

if copy:
    print("Copying Validation files...")
    for index, row in mavd.iterrows():
         # access data using column names
        audio_fname = row['audio_filename'] 
        shutil.copy(os.path.join(mavd_path, "audio_segments", "validate", audio_fname), os.path.join(out_val_dir,audio_fname))

    print("Validation files copied")
    '''
    print("Copying test files...")
    for index, row in test.iterrows():
         # access data using column names
        audio_fname = row['audio_filename'] 
        shutil.copy(os.path.join(mavd_path, "audio_segments","test", audio_fname), os.path.join(out_test_dir,audio_fname))

    print("Test files copied")
    '''
    print("Copying train files (this may take a while)...")

    for index, row in sonyc.iterrows():
         # access data using column names
        audio_fname = row['audio_filename'] 
        shutil.copy(os.path.join(sonyc_audio_path, audio_fname), os.path.join(out_train_dir,audio_fname))

    print("Train files copied!")

    