conda activate ust
# Download VGGish model files
mkdir -p ../vggish
pushd ../vggish
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
popd

# Download SONYC
mkdir -p $SONYC_UST_PATH
pushd $SONYC_UST_PATH
wget https://zenodo.org/record/2590742/files/annotations.csv
wget https://zenodo.org/record/2590742/files/audio.tar.gz
wget https://zenodo.org/record/2590742/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/2590742/files/README.md

# Decompress audio
tar xf audio.tar.gz
rm audio.tar.gz
popd

#Download MAVD

mkdir -p $MAVD_PATH
pushd ../data_generation
python download_mavd.py $MAVD_PATH
popd
