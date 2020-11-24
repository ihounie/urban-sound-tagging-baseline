# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:13:30 2020

@author: pcancela
"""

import os
import csv
import numpy
import json
import matplotlib.pyplot as plt

plt.style.use('seaborn')

tags = ["motorcycle/engine_idling","motorcycle/engine_accelerating","car/engine_accelerating","car/engine_idling","1-2_medium-sounding-engine_presence","bus/engine_accelerating","bus/engine_idling","truck/engine_accelerating","truck/engine_idling","chatter","music"]



def hist_label(file_path,tag, segments):
    verbose = 0
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        gap_start = 0;
        for row in csv_reader:
            if verbose:
                print(f'\t{row[0]} {row[1]} {row[2]}')
            time_start = float(row[0])
            time_end = float(row[1])
            if(row[2] == tag):
                gap_end = time_start;
                segments.append(time_end-time_start)

    return segments





main_counter = 1;
for tag in tags:
    largos =[]
    for dirpath, dnames, fnames in os.walk("../datasets/MAVD/annotations_train/"):
        for f in fnames:
            if f.endswith(".txt") and not f.startswith('._'):
                main_counter += 1
                print(os.path.join(dirpath, f))
                name_base = "audio_"+str(main_counter)
                largos = hist_label(os.path.join(dirpath, f),tag,largos)
                print(largos)
    plt.figure(figsize=(4,3))   
    #plt.grid(b=None)
    plt.xlabel("Duración (s)")
    plt.ylabel("Cantidad de Instancias")
    plt.title(tag.replace("/", ""))
    plt.hist(largos)
    plt.savefig("../figures/datasets/"+tag.replace("/", "")+"_count.png", dpi=300)
    
        


