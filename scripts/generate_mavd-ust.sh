conda activate ust
pushd ../data_generation
python generate_mavd-ust.py $MAVD_PATH $MAVD_UST_PATH -c -e
popd