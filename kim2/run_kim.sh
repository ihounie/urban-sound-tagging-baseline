# Set path
SONYC_UST_PATH=/home/hounie/audio/urban-sound-tagging-baseline/mavd-ust

#python3 extract_mel.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data mels
python3 train.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml mels checkpoints validation_output
python3 evaluate.py $SONYC_UST_PATH/annotations.csv validation_output/output_max.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml