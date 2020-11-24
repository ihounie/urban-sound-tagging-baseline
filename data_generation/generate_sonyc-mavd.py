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
import argparse

def copy_all_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sonyc_path", type=str, default="/clusteruy/home/ihounie/SONYC")
    parser.add_argument("mavd_path", type=str, default="/clusteruy/home/ihounie/MAVD")
    parser.add_argument("output_path", type=str, default="/clusteruy/home/ihounie/urban-sound-tagging-baseline/sonyc-mavd/data")
    parser.add_argument("--rootdir_path", type=str, default='../')
    parser.add_argument('-c','--copy', action='store_true', help="copy files on output dir")
    parser.add_argument('-e','--extract', action='store_true', help="generate jsons from mavd")
    parser.add_argument('-d','--duration', type=float, default=10.0, help="audio segment duration in seconds")
    parser.add_argument('-ml','--min_label', type=float, default=3.0, help="minimum event length of considered labels")
    parser.add_argument('-hop','--hop_size', type=float, default=1.0, help="hop for extracting overlapping train segments")
    parser.add_argument('-tl','--tagless', type=float, default=2.0, help="increase to extract more tagless segments to improve class balance")
    

    args = parser.parse_args()
    rootdir_path = args.rootdir_path
    mavd_path = args.mavd_path
    sonyc_path = args.sonyc_path
    output_path = args.output_path 
    sonyc_audio_path = os.path.join(sonyc_path, "train")

    
    if args.extract:
        process_mavd(duration = args.duration, #seconds,
                     clip_length = args.duration,
                     min_label = args.min_label,
                     overlap  = args.hop_size,
                    mavd_path = mavd_path, split="train", train_frac = 0.5, tagless_frac = args.tagless )
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

    if args.copy:
        print("Copying Test files...")
        for index, row in mavd.iterrows():
             # access data using column names
            audio_fname = row['audio_filename'] 
            shutil.copy(os.path.join(mavd_path, "audio_segments", "validate", audio_fname), os.path.join(out_val_dir,audio_fname))
        print("Test files copied")

        print("Copying train files (this may take a while)...")

        for index, row in sonyc.iterrows():
             # access data using column names
            audio_fname = row['audio_filename'] 
            shutil.copy(os.path.join(sonyc_audio_path, audio_fname), os.path.join(out_train_dir,audio_fname))

        print("Train files copied!")

    
