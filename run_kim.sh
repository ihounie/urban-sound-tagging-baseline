# Set path
MAVD_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/mavd-ust
SONYC_UST_PATH=/clusteruy/home/ihounie/urban-sound-tagging-baseline/sonyc-mavd

# TRAIN AND EVALUATE ON MAVD

#python3 extract_mel.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data $MAVD_UST_PATH/data/features/mels
#python3 train.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/kim/coarse/checkpoints $MAVD_UST_PATH/output/kim/coarse/validation_output
#python3 evaluate.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/kim/coarse/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml kim mavd
python3 train-fine.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/kim/fine/checkpoints $MAVD_UST_PATH/output/kim/fine/validation_output --num_epochs 2
python3 evaluate-fine.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/kim/fine/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml kim mavd

# TRAIN ON SONYC AND EVALUATE ON MAVD
#python3 extract_mel.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data $SONYC_UST_PATH/data/features/mels
#python3 train.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/features/mels $SONYC_UST_PATH/features/kim/coarse/checkpoints $SONYC_UST_PATH/output/kim/coarse/validation_output
#python3 evaluate.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/output/kim/coarse/validation_output/output_max.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml kim sonyc
#python3 train-fine.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/features/mels $SONYC_UST_PATH/features/kim/fine/checkpoints $SONYC_UST_PATH/output/kim/fine/validation_output
#python3 evaluate-fine.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/output/kim/fine/validation_output/output_max.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml kim sonyc

# FINE-TUNE
#python3 extract_mel.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data $MAVD_UST_PATH/data/features/mels
WEIGHT_DIR_COARSE=$SONYC_UST_PATH/features/kim/coarse/checkpoints
WEIGHT_DIR_FINE=$SONYC_UST_PATH/features/kim/fine/checkpoints
WEIGHT_FILE_COARSE=$(ls -t $WEIGHT_DIR_COARSE/* | head -n 1)
WEIGHT_FILE_FINE=$(ls -t $WEIGHT_DIR_FINE/* | head -n 1)
echo "Fine tuning, label mode: COARSE"
echo "Using $WEIGHT_FILE_COARSE pre-trained weights"

echo "python3 fine_tune.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/kim/fine_tune/coarse/checkpoints $MAVD_UST_PATH/output/kim/fine_tune/coarse/validation_output ${WEIGHT_DIR_COARSE}/${WEIGHT_FILE_COARSE} --labels coarse"
#python3 fine_tune.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/kim/fine_tune/coarse/checkpoints $MAVD_UST_PATH/output/kim/fine_tune/coarse/validation_output ${WEIGHT_FILE_COARSE} --label_mode coarse --freeze 6

#python3 evaluate.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/kim/fine_tune/coarse/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml kim fineTune6
echo "Fine tuning, label mode: FINE"
echo "Using $WEIGHT_FILE_FINE pre-trained weights"

#python3 fine_tune.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml $MAVD_UST_PATH/features/mels $MAVD_UST_PATH/features/kim/fine_tune/fine/checkpoints $MAVD_UST_PATH/output/kim/fine_tune/fine/validation_output ${WEIGHT_FILE_FINE} --label_mode fine --freeze 6 --num_epochs 2
#python3 evaluate-fine.py $MAVD_UST_PATH/data/annotations.csv $MAVD_UST_PATH/output/kim/fine_tune/fine/validation_output/output_max.csv $MAVD_UST_PATH/data/dcase-ust-taxonomy.yaml kim fineTune6
