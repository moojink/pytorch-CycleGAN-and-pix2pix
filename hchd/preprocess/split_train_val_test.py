from os import listdir
from shutil import copyfile
from tqdm import tqdm
import os
import random

"""Randomly splits human/robot datasets into 90%-5%-5% train-val-test."""
for ee_type, domain in [('human', 'A'), ('robot', 'B')]: # end effector type
    images_dir = f'images/{ee_type}/'
    image_files = [f for f in listdir(images_dir)]
    random.shuffle(image_files)
    num_images = len(image_files)
    num_train_images = int(num_images * 0.9)
    num_val_images = int(num_images * 0.05)
    train_images = image_files[:num_train_images]
    val_images = image_files[num_train_images:num_train_images+num_val_images]
    test_images = image_files[num_train_images+num_val_images:]
    for split_type, image_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
        output_images_dir = f'images/{split_type}{domain}/' # CycleGAN data dirs must be named trainA, trainB, valA, valB, testA, testB
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
        for image_filename in tqdm(image_list):
            copyfile(images_dir + image_filename, output_images_dir + image_filename)
