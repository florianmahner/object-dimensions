import glob
import os
import shutil
import random
import numpy as np
import sys
import subprocess

"""
This script downloads the things and things plus image datasets from 
https://osf.io/6kzat/ and https://osf.io/ce9km/ and unzips them.

It merges the things plus images into the things images by renaming each
things plus image to <object_name>_plus.jpg and moving it to the object folder
of the things images.

This script also downloads the things behavior triplets from and creates
a train val split of the data from the provided trainset.txt file.
"""

# Change this to a preferred locations
FOLDER = "./data_figshare/images/things"


# Check if folder is empty whether to start fresh
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
    print(f"Created folder {FOLDER}")


# else:
#     print("Skipping download, folder {} already exists.".format(FOLDER))
#     # shutil.rmtree(FOLDER)
#     sys.exit(0)

# print("Downloading things data (around 5gb, takes a while)...")
wget_commands = [
    #     [
    #         "wget",
    #         "https://osf.io/download/6kzat/",
    #         "-O",
    #         "{}/A-C.zip".format(FOLDER),
    #     ],
    #     [
    #         "wget",
    #         "https://osf.io/download/whm5p/",
    #         "-O",
    #         "{}/D-K.zip".format(FOLDER),
    #     ],
    #     [
    #         "wget",
    #         "https://osf.io/download/ay8mv/",
    #         "-O",
    #         "{}/L-Q.zip".format(FOLDER),
    #     ],
    #     [
    #         "wget",
    #         "https://osf.io/download/qm6gh/",
    #         "-O",
    #         "{}/R-S.zip".format(FOLDER),
    #     ],
    #     [
    #         "wget",
    #         "https://osf.io/download/c2spb/",
    #         "-O",
    #         "{}/T-Z.zip".format(FOLDER),
    #     ],
    [
        "wget",
        "https://osf.io/download/j3mn2/",
        "-O",
        "{}/plus.zip".format(FOLDER),
    ],
]

for cmd in wget_commands:
    subprocess.run(cmd)

breakpoint()

password = "things4all"
print("Unzipping things data... takes a while...")
for file_path in glob.glob(os.path.join(FOLDER, "*.zip")):
    cmd = f"unzip -q -P {password} {file_path} -d {os.path.join(FOLDER, 'images')}"
    subprocess.run(cmd.split())

print("Renaming plus files to *_plus.jpg... and moving them to the object folder")

breakpoint()
for file_path in glob.glob(os.path.join(FOLDER, "images", "images_resized", "*.jpg")):
    base = os.path.splitext(os.path.basename(file_path))[0]
    new_file_path = os.path.join(FOLDER, f"{base}_plus.jpg")
    os.rename(file_path, new_file_path)

# # Somehow these are not expanded correctly
for file_path in glob.glob(os.path.join(FOLDER, "object_images_L-Q", "*")):
    shutil.move(file_path, os.path.join(FOLDER))

shutil.rmtree(os.path.join(FOLDER, "object_images_L-Q"))
shutil.rmtree(os.path.join(FOLDER, "__MACOSX"))
for file_path in glob.glob(os.path.join(FOLDER, "**/*.zip"), recursive=True):
    os.remove(file_path)

for file in os.listdir(f"{FOLDER}/images"):
    if file.endswith("_plus.jpg"):
        base = file.split("_plus.jpg")[0]
        os.rename(f"{FOLDER}/images/{file}", f"{FOLDER}/{base}/{file}")
