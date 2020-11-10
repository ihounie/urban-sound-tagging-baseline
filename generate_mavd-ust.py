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
import pathlib

import sys

def copy_all_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mavd_path", type=str, default="/clusteruy/home/ihounie/MAVD")
    parser.add_argument("output_path", type=int, default="/clusteruy/home/ihounie/urban-sound-tagging-baseline/sonyc-mavd/data")
    parser.add_argument('-c','--copy', action='store_true', help="copy files on output dir")
    parser.add_argument('-e','--extract', action='store_true', help="generate jsons from mavd")
    parser.add_argument('-d','--duration', type=float, default=10.0, help="audio segment duration in seconds")
    parser.add_argument('-ml','--min_label', type=float, default=3.0, help="minimum event length of considered labels")
    parser.add_argument('-hop','--hop_size', type=float, default=1.0, help="hop for extracting overlapping train segments")
    parser.add_argument('-tl','--tagless', type=float, default=2.0, help="increase to extract more tagless segments to improve class balance")
  
    args = parser.parse_args()
    mavd_path = args.mavd_path
    output_path = args.output_path


    if args.extract:
        print("extracting audios from mavd")
        process_mavd(duration = 10, #seconds,
                     clip_length = 10,
                     min_label = 3,
                     overlap  = 1,
                    mavd_path = mavd_path, split="train", train_frac = 0.5, tagless_frac = 2 )
        jsons2csv(mode="fine",mavd_path = mavd_path, split="train", split_out ="train")
        jsons2csv(mode="fine",mavd_path = mavd_path, split="train", split_out ="validate")
    annot_csv_pth_train = os.path.join(mavd_path, "train","annotations.csv")
    train =  pd.read_csv(annot_csv_pth_train)
    print("*"*20)
    print(f" {len(train.index)} train samples")
    print("*"*20)

    annot_csv_pth_val = os.path.join(mavd_path, "validate","annotations.csv")
    val =  pd.read_csv(annot_csv_pth_val)
    print("*"*20)
    print(f" {len(val.index)} val samples")
    print("*"*20)
    '''
    if False:
        process_mavd(duration = 10, #seconds,
                     clip_length = 10,
                     min_label = 5,
                     overlap  = 10,
                    mavd_path = mavd_path, split="test" )
        jsons2csv(mode="fine",mavd_path = mavd_path, split = "test", split_out="test" )
        annot_csv_pth_test = os.path.join(mavd_path, "test","annotations.csv")
        test =  pd.read_csv(annot_csv_pth_test)
    '''
    assert(len(list(set(train.columns) - set(val.columns)))==0)
    assert(len(list(set(val.columns) - set(train.columns)))==0)

    merged = pd.concat([train, val])
    print(f"Total samlples: {len(merged.index)}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(os.getcwd())
    merged.to_csv(os.path.join(output_path,'annotations.csv'), index=False)
    #############################################################
    # Another annotations file with test files as validation set
    ############################################################
    #test["split"] = "validate"
    #merged_test = pd.concat([train, test])
    #merged_test.to_csv(os.path.join(output_path,'annotations_test.csv'), index=False)

    #out_test_dir = os.path.join(output_path,"test")
    out_train_dir = os.path.join(output_path,"train")
    out_val_dir = os.path.join(output_path,"validate")

    #pathlib.Path(out_test_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_val_dir).mkdir(parents=True, exist_ok=True)
    if args.copy:
        print("Copying Validation files...")
        for index, row in val.iterrows():
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

        for index, row in train.iterrows():
             # access data using column names
            audio_fname = row['audio_filename'] 
            shutil.copy(os.path.join(mavd_path, "audio_segments","train", audio_fname), os.path.join(out_train_dir,audio_fname))

        print("Train files copied!")
