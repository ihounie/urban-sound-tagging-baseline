import zipfile
import wget
import os
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mavd_path", type=str, default="/clusteruy/home/ihounie/MAVD",
                        help="path to store MAVD dataset")
  
    args = parser.parse_args()

MAVD_dir = args.mavd_path
zenodo_files = ['audio_train.zip', 'annotations_train.zip']
MAVD_files = []
for f in zenodo_files:
    file = os.path.join(MAVD_dir, f.split(".")[0])
    if not os.path.exists(file):
        print(f"{file} not found on path")
        # name of the audio files in the dataset
        MAVD_files.append(f)

# url of the MAVD dataset in zenodo
MAVD_url = 'https://zenodo.org/record/3338727/files/'


for zip_file in MAVD_files:
    print('Downloading file: ', zip_file)
    wget.download(MAVD_url + zip_file, MAVD_dir)
    print()
print('Done!')

for zip_file in MAVD_files:
    print('Extracting file: ', zip_file)
    zip_ref = zipfile.ZipFile(os.path.join(MAVD_dir,zip_file)) # create zipfile object
    zip_ref.extractall(MAVD_dir) # extract file to dir
    zip_ref.close() # close file
    os.remove(os.path.join(MAVD_dir,zip_file)) # delete zipped file
print('Done!')