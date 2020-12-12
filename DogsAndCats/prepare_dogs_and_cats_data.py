import os
import tensorflow as tf
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
"""
Filter out corrupted images
https://colab.research.google.com/github/keras-team/keras-io
/blob/master/examples/vision/ipynb/image_classification_from_scratch.
ipynb#scrollTo=pZkc-r_rR3n7
Iš nuorodos atsisiunčiami duomenis ir nukraunami į PetImages
"""

# Atsikratome nuo blogų foto
num_skipped = 0
home_folder = 'C:/DI/Data/PetImages/'
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(home_folder, folder_name)
    for fname in listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

    print("Deleted %d images" % num_skipped)

# Sukūriame reikalingą subfolderių struktūrą
# Programa iš Deep Learning for Computer Vision,
# Jason Brownlee, 2019, Chapter 21
#
# create directories
dataset_home = 'C:/DI/Data/dataset_dog_cat/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['Dog/', 'Cat/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

# Perrašome iš originalaus Dogs and Cats duomenų masyvo į sukurtus
# folderius, sumažindami duomenų kiekį, kartu sukurdami train ir test
# duomenų masyvus

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
num_max = 1000  # maksimalus duomenų kiekis vienoje kategorijoje
# copy training dataset images into subdirectories
for folder_name in ("Cat", "Dog"):
    src_directory = os.path.join(home_folder, folder_name)
    num = 0
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        dst = dataset_home + dst_dir + folder_name + '/' + file
        copyfile(src, dst)
        num += 1
        if (num >= num_max):
            break
    print(src_directory, " ", len(listdir(src_directory)))
    dst_directory = dataset_home + 'train/' + folder_name
    print(dst_directory, " ", len(listdir(dst_directory)))
    dst_directory = dataset_home + 'test/' + folder_name
    print(dst_directory, " ", len(listdir(dst_directory)))
