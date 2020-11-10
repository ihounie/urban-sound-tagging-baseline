# Set path
MAVD_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/mavd-ust
SONYC_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/sonyc-mavd

# TRAIN AND EVALUATE ON MAVD

#python3 extract_mel.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data $MAVD_UST_PATH/data/features/mels
python3 train.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/rf_le/coarse/checkpoints $MAVD_UST_PATH/output/rf_le/coarse/validation_output --label_mode coarse --latent_dim=1 --knn=3
python3 evaluate.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/rf_le/coarse/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml rf_le mavd
#python3 train.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/rf_le/fine/checkpoints $MAVD_UST_PATH/output/rf_le/fine/validation_output --label_mode fine --latent_dim=1 --knn=1
#python3 evaluate-fine.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/rf_le/fine/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml rf_le mavd

