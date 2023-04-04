import glob
import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm


""" Reduces the THINGS image database to have a balanced dataset of N samples that can 
be specified. We extract in total N samples per class. This means N samples per class minus 1
plus the behavior collected images. The total amount of images is thus (N-1) * 1854 + 1854 (behavior)
"""


SAMPLES_PER_CLASS = 12
BASE_DIR = "/home/florian/THINGS/image_data/images"
NEW_DIR = BASE_DIR + str(SAMPLES_PER_CLASS)
START_FRESH = True


if START_FRESH:
    # Copy all contents and then parse
    shutil.rmtree(NEW_DIR)
    shutil.copytree(BASE_DIR, NEW_DIR)

files = glob.glob(NEW_DIR + "/*/*", recursive=True)
random.seed(42)
random.shuffle(files)
files_per_class = defaultdict(int)

for f in tqdm(files):
    class_name = f.split("/")[-2]
    files_per_class[class_name] += 1

    if "b.jpg" in f:
        os.remove(f)
        files_per_class[class_name] -= 1  # remove from count

    # we  add reference images at the end, so we want only 11 images per class without the ref. image
    elif files_per_class[class_name] > (SAMPLES_PER_CLASS - 1):
        os.remove(f)


ref_images = "../reference_images"
ref_images = glob.glob(ref_images + "/*", recursive=True)

for f in ref_images:
    name = os.path.split(f)[1]
    name, ext = name.split(".")
    name_new = name + "/" + name + "_01b." + ext
    name_new = os.path.join(NEW_DIR, name_new)
    shutil.copy(f, name_new)
