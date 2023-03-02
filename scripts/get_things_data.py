import glob
import os
import shutil
import random
import numpy as np
import sys

# Change this to a preferred locations
FOLDER = "./data/things"

# Check if folder is empty whether to start fresh
if glob.glob(os.path.join(FOLDER, "*")):
    print(f"FOLDER {FOLDER} is not empty. Skipping download.")
    reply = input("Do you want to delete the folder and download the data again? [y/n] ")

    # If yes, delete the folder
    if reply.lower() == "y":
        print(f"Deleting {FOLDER} and creating new folder...")
        shutil.rmtree(FOLDER)
        os.makedirs(FOLDER)
    else:
        print("Skipping download.")
        sys.exit(0)

print("Downloading things data (around 5gb, takes a while)...")
wget_commands = [
    "wget https://osf.io/download/6kzat/ -O {}/images/A-C.zip".format(FOLDER),
    "wget https://osf.io/download/whm5p/ -O {}/images/D-K.zip".format(FOLDER),
    "wget https://osf.io/download/ay8mv/ -O {}/images/L-Q.zip".format(FOLDER),
    "wget https://osf.io/download/qm6gh/ -O {}/images/R-S.zip".format(FOLDER),
    "wget https://osf.io/download/c2spb/ -O {}/images/T-Z.zip".format(FOLDER),
    "wget https://osf.io/download/ce9km/ -O {}/plus.zip".format(FOLDER),
    "wget https://files.osf.io/v1/resources/f5rn6/providers/osfstorage/62d6b34bc79a4c38719e62fa/?view_only=3eca2c272c4643e5b7c3f9a875fa9dc7&zip= -O {}/triplets/behavior_triplets.zip".format(FOLDER)
]

for cmd in wget_commands:
    os.system(cmd)

password = "things4all"
print("Unzipping things data... takes a while...")
for file_path in glob.glob(os.path.join(FOLDER, "images", "*.zip")):
    os.system(f"unzip -q -P {password} {file_path} -d {os.path.join(FOLDER, 'images')}")

os.system(
    f"unzip -q {os.path.join(FOLDER, 'triplets', 'behavior_triplets.zip')} -d {os.path.join(FOLDER, 'triplets')}"""
)

print("Renaming plus files to *_plus.jpg... and moving them to the object folder")
for file_path in glob.glob(os.path.join(FOLDER, "images", "images_resized", "*.jpg")):
    base = os.path.splitext(os.path.basename(file_path))[0]
    new_file_path = os.path.join(FOLDER, "images", f"{base}_plus.jpg")
    os.rename(file_path, new_file_path)

# # Somehow these are not expanded correctly
for file_path in glob.glob(os.path.join(FOLDER, "images", "object_images_L-Q", "*")):
    shutil.move(file_path, os.path.join(FOLDER, "images"))

shutil.rmtree(os.path.join(FOLDER, "images", "object_images_L-Q"))
shutil.rmtree(os.path.join(FOLDER, "images", "__MACOSX"))
for file_path in glob.glob(os.path.join(FOLDER, "**/*.zip"), recursive=True):
    os.remove(file_path)

for file in os.listdir(f"{FOLDER}/images"):
    if file.endswith("_plus.jpg"):
        base = file.split("_plus.jpg")[0]
        os.rename(f"{FOLDER}/images/{file}", f"{FOLDER}/images/{base}/{file}")


# Create a train val split of the data in things. This is due to a misnomer
# of the original data. The validation set is actually a split of the trainset.txt
train_split = 0.9
data = np.loadtxt(os.path.join(FOLDER, "triplets", "trainset.txt"))
random.seed(0)
random.shuffle(data)
split_idx = int(train_split * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]
np.savetxt(os.path.join(FOLDER, "triplets", "train_90.txt"), train_data)
np.savetxt(os.path.join(FOLDER, "triplets", "test_10.txt"), val_data)