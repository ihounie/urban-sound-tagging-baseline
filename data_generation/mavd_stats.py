#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:33:07 2020

@author: root
"""

import pandas as pd
from matplotlib import pyplot as plt
from process_jsons import *
import pathlib
import os

plt.style.use('seaborn')




min_labels = [3, 3]
overlaps = [10, 1]
mavd_path = "../datasets/MAVD"
process=True#Set to false to disable processing MAVD (debugging purposes, maybe)
for min_label, overlap in zip(min_labels, overlaps):
    pathlib.Path(os.path.join(mavd_path, "train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(mavd_path, "validate")).mkdir(parents=True, exist_ok=True)
    if process:
        process_mavd(duration = 10, #seconds,
                 clip_length = 10,
                 min_label = min_label,
                 overlap  = overlap,
                mavd_path = mavd_path, split="train", train_frac = 0.5, tagless_frac = 2)
        jsons2csv(mode="fine",mavd_path = mavd_path, split="train", split_out ="train", audio=False)
        jsons2csv(mode="fine",mavd_path = mavd_path, split="train", split_out ="validate", audio=False)
    annot_csv_pth_train = os.path.join(mavd_path, "train","annotations.csv")
    annot_csv_pth_train = os.path.join(mavd_path, "train","annotations.csv")
    train =  pd.read_csv(annot_csv_pth_train)
    print("*"*20)
    print(f" {len(train.index)} train samples")
    print("*"*20)

    annot_csv_pth_val = os.path.join(mavd_path, "validate","annotations.csv")
    val =  pd.read_csv(annot_csv_pth_val)
    print("*"*20)
    print(f" {len(val.index)} test samples")
    print("*"*20)
    assert(len(list(set(train.columns) - set(val.columns)))==0)
    assert(len(list(set(val.columns) - set(train.columns)))==0)
    
    for split, data in zip (["train", "test"],[train, val]):
        n_files = len(data.index)
        sincero = data.drop(["split", "sensor_id", "audio_filename", "annotator_id"], axis=1)
        sincero = sincero[sincero > 0].sum()
        sincero = sincero[sincero > 0]
        plt.figure(figsize=(4,3))
        plt.bar([s.split("_")[0] for s in sincero.index][:3], sincero.values[:3]/n_files)
        #plt.grid(b=None)
        plt.title(f"Level: fine, Split: {split}, minimum length: {min_label}s, hop: {overlap}s")
        plt.savefig(f"../figures/datasets/fine_{split}_min_label_{min_label}_hop_{overlap}.png", dpi=300)
        plt.show()
        plt.figure(figsize=(4,3))
        plt.bar([s.split("_")[0] for s in sincero.index][5:], sincero.values[5:]/n_files )
        #plt.grid(b=None)
        plt.title(f"Level: coarse, Split: {split}, minimum length: {min_label}s, hop: {overlap}s")
        plt.savefig(f"../figures/datasets/coarse_{split}_min_label_{min_label}_hop_{overlap}.png", dpi=300)
        plt.show()



