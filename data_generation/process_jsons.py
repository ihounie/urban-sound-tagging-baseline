# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:13:30 2020

@author: pcancela
"""

# Now go through each of the jsons and crop each audio segment
import sox
import ntpath
import os
import json
import zipfile
import wget
import numpy
import os
import pathlib

import csv

  

def create_json(file_path, name , main_sound_type, time_start, time_end,min_label):
    json_segment = { "name": name,"file_path": file_path, "main_label": main_sound_type, "time_start":time_start, "time_end":time_end}
    labels = [];
    verbose = True
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if verbose:
                print(f'\t{row[0]} {row[1]} {row[2]}')
            line_count += 1
            time_s = float(row[0])
            time_e = float(row[1])
            label = row[2]
        #    if (time_s <= time_start) and (time_end <= time_e): # REVISAR CRITERIO
            if((min(time_e,time_end)-max(time_s,time_start)) > min_label):
                labels.append(label)
        if verbose:            
            print(f'Processed {line_count} lines.')
    json_segment["labels"] = labels
    print(json_segment)
    return json_segment

def create_json_gap(file_path, name , time_start, time_end):
    json_segment = { "name": name,"file_path": file_path, "main_label": "music", "time_start":time_start, "time_end":time_end}
    labels = [];
    verbose = True
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if verbose:
                print(f'\t{row[0]} {row[1]} {row[2]}')
            line_count += 1
            time_s = float(row[0])
            time_e = float(row[1])
            label = row[2]
        #    if (time_s <= time_start) and (time_end <= time_e): # REVISAR CRITERIO
            if((min(time_e,time_end)-max(time_s,time_start)) > min_label):
                labels.append(label)
        if verbose:            
            print(f'Processed {line_count} lines.')
    json_segment["labels"] = labels
    print(json_segment)
    return json_segment


def procesar_anotaciones(file_path, name_base, sound_type, clip_length,min_label,overlap,mavd_path, split):
    """" Reads the annotations from MAVD on file_path, and writes jsons with a filename with prefix name_base.
        The
        clip_length (secs) - total length of the audio
        min_label (secs) - minimum length of the label
        overlap (secs) - maximum time overlap between clips generated from the same label.
    """

    verbose = 0
    segments = [];
    counter = 0
    annot_json_dir = os.path.join(mavd_path, split,"jsons")
    if not os.path.exists(annot_json_dir):
        pathlib.Path(annot_json_dir).mkdir(parents=True, exist_ok=True)
        #os.mkdir(annot_json_dir, exist_ok=True)
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if verbose:
                print(f'\t{row[0]} {row[1]} {row[2]}')
            line_count += 1
            time_start = float(row[0])
            time_end = float(row[1])
            if (time_end-time_start > min_label) and (row[2] == sound_type):
                print(f'\t{row[0]} {row[1]} {row[2]}')
                for sg_start_time in numpy.arange(max(time_start-(clip_length-min_label),0),time_end-min_label,overlap):
                    segments.append((sg_start_time,sg_start_time+clip_length))
                    segment_name = name_base+"_"+str(counter); # DECIDIR/MEJORAR
                    start = sg_start_time
                    end = numpy.around(sg_start_time+clip_length, decimals=3)
                    assert(numpy.around(end-start, decimals=6)==clip_length)
                    json_segment = create_json(file_path, segment_name , row[2] , sg_start_time, sg_start_time+clip_length,min_label)
                    with open(os.path.join(mavd_path, split,"jsons", segment_name+".json"), 'w') as outfile:
                        json.dump(json_segment, outfile)
                    counter += 1
                if verbose:    
                    print(segments)
        if verbose:            
            print(f'Processed {line_count} lines.')
    return segments, counter




# This is the main script that goes over all desired tags for a duration 
# looking for segments in all the annotation  MAVD files in the given path

d = 10
def process_mavd(duration = 10, #seconds,
                 clip_length = 10,
                 min_label = 1,
                 overlap  = 1,
                 mavd_path = "MAVD/", split="train", train_frac = 0.8, tagless_frac = 0.3):

    tags = ["motorcycle/engine_idling","motorcycle/engine_accelerating","car/engine_accelerating","car/engine_idling","1-2_medium-sounding-engine_presence","bus/engine_accelerating","bus/engine_idling","truck/engine_accelerating","truck/engine_idling","chatter","music"]
    
    main_counter=1
    dir_annot = os.path.join(mavd_path, "annotations_"+split+"/")
    print(f"fetching annotations from {dir_annot}")
    i=0
    for dirpath, dnames, fnames in os.walk(dir_annot):
        for f in fnames:
            if f.endswith(".txt") and not f.startswith('._'):
                train = numpy.random.binomial(1, train_frac, size=1)
                if train:
                    split_out = "train"
                    overlap_out = overlap
                else:
                    split_out = "validate"
                    overlap_out = clip_length
                print(split_out) 
                for tag in tags:
                    main_counter += 1
                    print(os.path.join(dirpath, f))
                    name_base = "audio_"+str(main_counter)
                    segments, counter = procesar_anotaciones(os.path.join(dirpath, f),name_base,tag,duration,min_label,overlap_out, mavd_path, split_out)
                    print(segments)
                    # Adjust number of tagless segments
                    if train:
                        n_gaps = tagless_frac*counter
                        gaps = find_gaps(os.path.join(dirpath, f),  name_base,clip_length, n=n_gaps, mavd_path = mavd_path, split = split_out)
                    else:
                        n_gaps = max(0,(tagless_frac-0.1)*counter)
                i+=1
    if i==0:
        print("No annotation files found")
                    
                    
def find_gaps(file_path, name_base,clip_length, n=10, mavd_path = "MAVD", split = "train"):
    tags = ["motorcycle/engine_idling",
                    "motorcycle/engine_accelerating",
                    "car/engine_accelerating",
                    "car/engine_idling",
                    "bus/engine_accelerating",
                    "bus/engine_idling",
                    "truck/engine_accelerating",
                    "truck/engine_idling"] 
    verbose = 0
    segments = [];
    counter = 0
    annot_json_dir = os.path.join(mavd_path, split,"jsons")
    if not os.path.exists(annot_json_dir):
        os.pathlib.Path(annot_json_dir).mkdir(parents=True, exist_ok=True)
    
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        data = []
        for row in csv_reader:
            data.append([float(row[0]),float(row[1]),row[2]])        
        data.sort()
        gap_start = 0;
        for row in data:
            if verbose:
                print(f'\t{row[0]} {row[1]} {row[2]}')
            line_count += 1
            time_start = float(row[0])
            time_end = float(row[1])
            if((row[2] in tags) and time_start > gap_start):
                gap_end = time_start;
                if gap_end-gap_start>clip_length:
                    segments.append((gap_start,gap_start+clip_length))
                gap_start = time_end;
                segment_name = name_base+"_gap_"+str(counter);
                json_segment = { "name": segment_name,"file_path": file_path, "main_label": [], "time_start": gap_start, "time_end":gap_start+clip_length}
                json_segment["labels"] = []
                if counter<n:
                    with open(os.path.join(mavd_path, split,"jsons", segment_name+".json"), 'w') as outfile:
                        json.dump(json_segment, outfile)
                        counter += 1
                else:
                    print("too much gaps")
                    break
            else:
                if((row[2] in tags)):
                    gap_start = time_end            
                
    return segments


def jsons2csv( output_path = "./MAVD/audio_segments/",mavd_path = "MAVD/", split="train", mode = "coarse", split_out = "train", audio=True):
    
    correspondencias = [["motorcycle/engine_idling","1-1_small-sounding-engine_presence"],
                    ["motorcycle/engine_accelerating","1-1_small-sounding-engine_presence"],
                    ["car/engine_accelerating","1-2_medium-sounding-engine_presence"],
                    ["car/engine_idling","1-2_medium-sounding-engine_presence"],
                    ["bus/engine_accelerating","1-3_large-sounding-engine_presence"],
                    ["bus/engine_idling","1-3_large-sounding-engine_presence"],
                    ["truck/engine_accelerating","1-3_large-sounding-engine_presence"],
                    ["truck/engine_idling","1-3_large-sounding-engine_presence"],
                    ["chatter","7-1_person-or-small-group-talking_presence"],
                    ["music","6-X_music-from-uncertain-source_presence"],
                    ["dummy", "1-X_engine-of-uncertain-size_presence"],
                    ["dummy2", "7-X_other-unknown-human-voice_presence"],
                    ["dummy3","7-2_person-or-small-group-shouting_presence"],
                    ["dummy4","7-3_large-crowd_presence"],
                    ["dummy5","7-4_amplified-speech_presence"],
                    ["dummy6","6-1_stationary-music_presence"],
                    ["dummy7","6-2_mobile-music_presence"],
                    ["dummy8","6-3_ice-cream-truck_presence"]]

    correspondencias_coarse = [["motorcycle/engine_idling","1_engine_presence"],
                    ["motorcycle/engine_accelerating","1_engine_presence"],
                    ["car/engine_accelerating","1_engine_presence"],
                    ["car/engine_idling","1_engine_presence"],
                    ["bus/engine_accelerating","1_engine_presence"],
                    ["bus/engine_idling","1_engine_presence"],
                    ["truck/engine_accelerating","1_engine_presence"],
                    ["truck/engine_idling","1_engine_presence"],
                    ["chatter","7_human-voice_presence"],
                    ["music","6_music_presence"]]
    
          
    dict_corr = {}
    considered_labels = []          
    considered_labels_coarse = [] 
    
    for i in range(len(correspondencias)):    
        considered_labels.append(correspondencias[i][1])
        
    for i in range(len(correspondencias_coarse)):    
        considered_labels_coarse.append(correspondencias_coarse[i][1])
                    
                    
    columnas  = ["split","sensor_id","audio_filename","annotator_id",
    "1-1_small-sounding-engine_presence",
    "1-2_medium-sounding-engine_presence",
    "1-3_large-sounding-engine_presence",
    "1-X_engine-of-uncertain-size_presence",
    "2-1_rock-drill_presence",
    "2-2_jackhammer_presence",
    "2-3_hoe-ram_presence",
    "2-4_pile-driver_presence",
    "2-X_other-unknown-impact-machinery_presence",
    "3-1_non-machinery-impact_presence",
    "4-1_chainsaw_presence",
    "4-2_small-medium-rotating-saw_presence",
    "4-3_large-rotating-saw_presence",
    "4-X_other-unknown-powered-saw_presence",
    "5-1_car-horn_presence",
    "5-2_car-alarm_presence",
    "5-3_siren_presence",
    "5-4_reverse-beeper_presence",
    "5-X_other-unknown-alert-signal_presence",
    "6-1_stationary-music_presence",
    "6-2_mobile-music_presence",
    "6-3_ice-cream-truck_presence",
    "6-X_music-from-uncertain-source_presence",
    "7-1_person-or-small-group-talking_presence",
    "7-2_person-or-small-group-shouting_presence",
    "7-3_large-crowd_presence",
    "7-4_amplified-speech_presence",
    "7-X_other-unknown-human-voice_presence",
    "8-1_dog-barking-whining_presence",
    "1-1_small-sounding-engine_proximity",
    "1-2_medium-sounding-engine_proximity",
    "1-3_large-sounding-engine_proximity",
    "1-X_engine-of-uncertain-size_proximity",
    "2-1_rock-drill_proximity",
    "2-2_jackhammer_proximity",
    "2-3_hoe-ram_proximity",
    "2-4_pile-driver_proximity",
    "2-X_other-unknown-impact-machinery_proximity",
    "3-1_non-machinery-impact_proximity",
    "4-1_chainsaw_proximity",
    "4-2_small-medium-rotating-saw_proximity",
    "4-3_large-rotating-saw_proximity",
    "4-X_other-unknown-powered-saw_proximity",
    "5-1_car-horn_proximity",
    "5-2_car-alarm_proximity",
    "5-3_siren_proximity",
    "5-4_reverse-beeper_proximity",
    "5-X_other-unknown-alert-signal_proximity",
    "6-1_stationary-music_proximity",
    "6-2_mobile-music_proximity",
    "6-3_ice-cream-truck_proximity",
    "6-X_music-from-uncertain-source_proximity",
    "7-1_person-or-small-group-talking_proximity",
    "7-2_person-or-small-group-shouting_proximity",
    "7-3_large-crowd_proximity",
    "7-4_amplified-speech_proximity",
    "7-X_other-unknown-human-voice_proximity",
    "8-1_dog-barking-whining_proximity",
    "1_engine_presence",
    "2_machinery-impact_presence",
    "3_non-machinery-impact_presence",
    "4_powered-saw_presence",
    "5_alert-signal_presence",
    "6_music_presence",
    "7_human-voice_presence",
    "8_dog_presence"]
    
    
    
    
    verbose = False

    audio_path = os.path.join(mavd_path,"audio_"+ split)
    #i=0
    jsons_path = os.path.join(mavd_path, split_out,"jsons") 
    #delete file if it exists
    output_path = os.path.join(mavd_path,"audio_segments", split_out)
    if not os.path.exists(output_path):
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    annot_csv_pth = os.path.join(mavd_path, split_out,"annotations.csv")
    if os.path.isfile(annot_csv_pth):
        os.remove(annot_csv_pth)
    fp = open(annot_csv_pth, 'w')
    with fp:
        writer = csv.writer(fp)    
        writer.writerow(columnas)
    
        for dirpath, dnames, fnames in os.walk(jsons_path):   
            for f in fnames:
                if f.endswith(".json") and not f.startswith('._'):
                    print("Processing: ",jsons_path+'/'+f)
                    with open(os.path.join(jsons_path,f)) as json_file:
                        data = json.load(json_file)
                        if verbose:
                            print("JSON contents: ",data)
                        audio_filename = os.path.join(audio_path, os.path.splitext(os.path.basename(data['file_path']))[0]+".flac")
                        # Get file duration
                        audio_file_duration = sox.file_info.duration(audio_filename)
                        # check if tags fit audio segment duration
                        if data["time_end"]<audio_file_duration:
                            assert(data["time_end"]-data["time_start"]-10.0<10e-7)
                            #print("filename "+ audio_filename)
                            if audio:
                                tfm = sox.Transformer()
                                tfm.trim(data["time_start"], data["time_end"])
                                tfm.set_output_format(rate=48000, bits=16)
                                #tfm.build_file(jsons_path+f, output_path+f+".wav")
                                file_output_path = os.path.join(output_path, data["name"]+".wav")
                                tfm.build_file( audio_filename, file_output_path)
                            if verbose:
                                print("Segment file: "+ ntpath.basename(data["file_path"]))
                            val_cols = []
                            for j in range(len(columnas)):
                                if columnas[j] in considered_labels or (columnas[j] in considered_labels_coarse):
                                    found = 0;
                                    for i in range(len(correspondencias)):
                                        if correspondencias[i][1] == columnas[j] and correspondencias[i][0] in data['labels']:       
                                            found = 1
                                    for i in range(len(correspondencias_coarse)):
                                        if correspondencias_coarse[i][1] == columnas[j] and correspondencias_coarse[i][0] in data['labels']:       
                                            found = 1
                                    if found:
                                        val_cols.append(1)  
                                    else:
                                        if (columnas[j] in considered_labels) or (columnas[j] in considered_labels_coarse):
                                            val_cols.append(0)
                                        else:
                                            val_cols.append(0)
                                else:
                                    val_cols.append(-1)
                            val_cols[3] = 0    
                            val_cols[2] = data["name"]+".wav"
                            val_cols[0] = split_out
                            #i+=1
                            #val_cols[0] = i
                            writer.writerow(val_cols)
                            if verbose:
                                print(val_cols)
                        elif verbose:
                                print(val_cols)
   