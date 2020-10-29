#!/usr/bin/env bash

# Set path
SONYC_UST_PATH=/home/hounie/audio/urban-sound-tagging-baseline/mavd-ust

# Extract embeddings
pushd urban-sound-tagging-baseline
python3 extract_embedding.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data $SONYC_UST_PATH/features $SONYC_UST_PATH/vggish

# Train fine-level model and produce predictions
python3 classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/features/vggish $SONYC_UST_PATH/output baseline_fine --label_mode fine

exp_folder=$(ls -td -- $SONYC_UST_PATH/output/baseline_fine/* | head -n 1)

# Evaluate model based on AUPRC metric
python3 evaluate_predictions.py $exp_folder/output_mean.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml baseline mavd-ust

# Train coarse-level model and produce predictions
python3 classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/features/vggish $SONYC_UST_PATH/output baseline_coarse --label_mode coarse

exp_folder=$(ls -td -- $SONYC_UST_PATH/output/baseline_coarse/* | head -n 1)

# Evaluate model based on AUPRC metric
python3 evaluate_predictions.py $exp_folder/output_mean.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml baseline_coarse mavd-ust

# Return to the base directory
popd
