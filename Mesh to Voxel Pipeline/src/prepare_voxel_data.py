# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 01:02:09 2024

@author: ken
"""


"""
This program processes 3D model MAT data in a given input path, 
converts it into voxel grids and saves it into an HDF5 file 
along with labels for training, testing, and validation datasets.
"""


import numpy as np
import h5py
import glob
import os
import scipy.io
from random import shuffle
import random
from sklearn.model_selection import train_test_split

import mat_to_voxel_converter 

SEED = 448
random.seed(SEED)

# Default path if user input is invalid
default_path = 'ModelNet40/ModelNet40_Mat'

# Ask the user to input the directory path
input_path = input(f"Please enter the directory path containing MAT data (Default Path is '{default_path}'): ")

# Validate the user input
if not os.path.isdir(input_path):
    print(f"Invalid directory. Using default path: {default_path}")
    input_path = default_path
    

label = {}
label_counter = 0

train_addrs = []
labels = []
combine_train = []
combine_test = []

# Iterate through files to collect and label data addresses.
for item in os.listdir(input_path):
    if item != '.DS_Store':
       
        if item not in label:
            # Assign the next number to the new item
            label[item] = label_counter
            # Increment the counter for the next item
            label_counter += 1       
        
        # Retrieve the file paths and labels for training and 
        # testing data and put them into lists for further
        # processing.
        current_label = label.get(item)
        current_dir = os.path.join(input_path, item)
        train_path = os.path.join(current_dir, 'train')
        test_path = os.path.join(current_dir, 'test')
        train_addrs = glob.glob(train_path + '/*.mat')
        test_addrs = glob.glob(test_path + '/*.mat')
        train_labels = np.full(len(train_addrs), current_label, dtype=int)
        print('Training data for %s: %d' % (item, len(train_addrs)))
        test_labels = np.full(len(test_addrs), current_label, dtype=int)
        print('Testing data for %s: %d' % (item, len(test_addrs)))
        temp_train_addrs = list(zip(train_addrs, train_labels))
        temp_test_addrs = list(zip(test_addrs, test_labels))
        combine_train += temp_train_addrs
        combine_test += temp_test_addrs

print(len(combine_test))
shuffle(combine_train)
shuffle(combine_test)

mat_train, train_label = zip(*combine_train)
mat_test, test_label = zip(*combine_test)

# split test data into test data and validation data
mat_test, mat_val, test_label, val_label = train_test_split(mat_test, test_label, test_size=0.5)

print('Total training example: %d' % (len(mat_train)))
print('Total testing example: %d' % (len(mat_test)))
print('Total validation example: %d' % (len(mat_val)))

# Create text files to store file paths and their associated labels 
#for the training, testing, and validation datasets for future use.
train_file = open('train40.txt', 'w')
test_file = open('test40.txt', 'w')
val_file = open('validation40.txt', 'w')

default_box_size = 32
try:
    # Ask the user to input a number
    box_size = int(float(input(f"Please enter the target voxel size (Default Value is {default_box_size}): ")))
except ValueError:
    # If there's any problem with the input (e.g., it's not a number), use 32 as the default value
    print(f"Invalid input. Using default box size value of '{default_box_size}'.")
    box_size = default_box_size
        
train_shape = (len(mat_train), box_size, box_size, box_size)
test_shape = (len(mat_test), box_size, box_size, box_size)
val_shape = (len(mat_val), box_size, box_size, box_size)

# Create hdf5 file and create datasets to store training,
# testing and validation datasets

# Default file name if user input is invalid
default_filename = "object40"

# Ask the user to input the hdf5 filename to use
try:
    hdf5_filename = input(f"Please enter the name of hdf5 file to save dataset to without \".htf5\" (Default file name is '{default_filename}'): ").strip()
    if not hdf5_filename:
        hdf5_filename = default_filename
    if hdf5_filename.endswith('.hdf5'):
        # Remove the '.hdf5' suffix
        hdf5_filename = hdf5_filename[:-5]
    # Validate the user input
except Exception as e:
    # If there's any problem, use the default filename
    print(f"Invalid file name. Using default file name '{default_filename}'.")
    hdf5_filename = default_filename

hdf5_filename += ".hdf5"
print(f"creating new hdf5 file: {hdf5_filename}")
hdf5_file = h5py.File(hdf5_filename, "w")
hdf5_file.create_dataset("train_mat", train_shape, np.int8)
hdf5_file.create_dataset("test_mat", test_shape, np.int8)
hdf5_file.create_dataset("val_mat", val_shape, np.int8)

hdf5_file.create_dataset("train_label", (len(train_label), 1), np.int8)
hdf5_file.create_dataset("test_label",  (len(test_label), 1), np.int8)
hdf5_file.create_dataset("val_label",   (len(val_label), 1), np.int8)

#voxels = np.zeros((box_size, box_size, box_size)) #32, 32, 32

# Convert each training MAT data into voxel grid and store in hdf5 file
for i in range(len(mat_train)):
    if i % 50 == 0:
        print('Training writing has finished: %d/%d' % (i, len(mat_train)))
    mat = scipy.io.loadmat(mat_train[i])
    vertices = mat['vertices']
    faces = mat['faces']
    
    train_voxels = mat_to_voxel_converter.convert_mat_to_voxel(vertices, faces, box_size)   
    
    if train_voxels is not None and train_voxels.size > 0:
        hdf5_file["train_mat"][i, ...] = train_voxels
        hdf5_file["train_label"][i] = train_label[i]
        train_file.write("%s, %d\n" % (mat_train[i], train_label[i]))
print('Training writing has finished...')

# Convert each testing MAT data into voxel grid and store in hdf5 file
for j in range(len(mat_test)):
    if j % 50 == 0:
        print('Testing writing has finished: %d/%d' % (j, len(mat_test)))
    mat = scipy.io.loadmat(mat_test[j])
    vertices = mat['vertices']
    faces = mat['faces']
    
    test_voxels = mat_to_voxel_converter.convert_mat_to_voxel(vertices, faces, box_size)
    
    if test_voxels is not None and test_voxels.size > 0:
        hdf5_file["test_mat"][j, ...] = test_voxels
        hdf5_file["test_label"][j] = test_label[j]
        test_file.write("%s, %d\n" % (mat_test[j], test_label[j]))
print('Testing writing has finished...')

# Convert each validation MAT data into voxel grid and store in hdf5 file
for k in range(len(mat_val)):
    if k % 50 == 0:
        print('Validation writing has finished: %d/%d' % (k, len(mat_val)))
    mat = scipy.io.loadmat(mat_val[k])
    vertices = mat['vertices']
    faces = mat['faces']
    
    val_voxels = mat_to_voxel_converter.convert_mat_to_voxel(vertices, faces, box_size)
    
    if val_voxels is not None and val_voxels.size > 0:
        hdf5_file["val_mat"][k, ...] = val_voxels
        hdf5_file["val_label"][k] = val_label[k]
        val_file.write("%s, %d\n" % (mat_val[k], val_label[k]))
print('Validation writing has finished...')
print("End of Program")