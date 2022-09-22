import glob
import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm


''' Gets the THINGS behavior immages givne *.b ending. Does not work right now. Not working 
right now! '''

BASE_DIR = '../../THINGS/image_data/images'
NEW_DIR = BASE_DIR + "_Behavior"
START_FRESH = True

files = glob.glob(BASE_DIR + '/*/*', recursive = True)


if START_FRESH:
    # Copy all contents and then parse
    shutil.rmtree(NEW_DIR)
    shutil.copytree(BASE_DIR, NEW_DIR)
