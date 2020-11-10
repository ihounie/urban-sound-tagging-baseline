# Set path
MAVD_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/mavd-ust
SONYC_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/sonyc-mavd

# TRAIN AND EVALUATE ON MAVD

#python3 extract_mel.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data $MAVD_UST_PATH/data/features/mels
#python3 train.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/rf/coarse/checkpoints $MAVD_UST_PATH/output/rf/coarse/validation_output --label_mode coarse
#python3 evaluate.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/rf/coarse/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml rf mavd
#python3 train.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/rf/fine/checkpoints $MAVD_UST_PATH/output/rf/fine/validation_output --label_mode fine
#python3 evaluate-fine.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/rf/fine/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml rf mavd

