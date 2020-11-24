conda activate ust
pushd ../data_generation
python generate_sonyc-mavd.py $SONYC_UST_PATH $MAVD_PATH $SONYC_MAVD_PATH -c
popd