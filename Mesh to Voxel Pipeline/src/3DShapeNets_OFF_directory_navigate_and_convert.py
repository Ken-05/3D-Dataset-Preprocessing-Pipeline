# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:56:58 2024

@author: ken
"""



"""
This script navigates from an input folder to all the lowest level
directories containing only off files and converts them to MAT files
and then stores the MAT files in the corresponding arrangement of 
directories in the output directory given.
"""


import os
import off_to_mat_converter


# Default path if user input is invalid
default_input_path = 'ModelNet40/ModelNet40_off'
default_output_path = 'ModelNet40/ModelNet40_Mat'

# Ask the user to input the directory paths
input_path = input(f"Please enter the directory path containing OFF data (Default Path is '{default_input_path}'): ")
output_path = input(f"Please enter the output directory path containing MAT data (Default Path is '{default_output_path}'): ")

# Validate the user inputs
if not os.path.isdir(input_path):
    print(f"Invalid input directory: {input_path}. Using default input path: {default_input_path}")
    input_path = default_input_path
    
if not os.path.isdir(output_path):
    print(f"Invalid output directory: {output_path}. Using default output path: {default_output_path}")
    out_path = default_output_path    



for item in os.listdir(input_path):
    if item != '.DS_Store' and item != 'README.txt' and item != '' and item != 'airplane' and item != 'bathtub' and item != 'bed' and item != 'bench' and item != 'bookshelf' and item != 'bottle' and item != 'bowl' and item != 'car' and item != 'chair' and item != 'cone' and item != 'cup' and item != 'curtain' and item != 'desk' and item != 'door' and item != 'dresser' and item != 'flower_pot' and item != 'glass_box' and item != 'guitar' and item != 'keyboard' and item != 'lamp' and item != 'laptop' and item != 'mantel' and item != 'monitor' and item != 'night_stand' and item != 'person' and item != 'piano':
    #if item != '.DS_Store' and item != 'README.txt':
        current_path = os.path.join(input_path, item)
        output_path = os.path.join(out_path, item)
        print(current_path)
        for file in os.listdir(current_path):
            if file != '.DS_Store':
                current_train_path = os.path.join(current_path, 'train')
                output_path_train = os.path.join(output_path, 'train')
                off_to_mat_converter.convert_off_to_mat(current_train_path, output_path_train)

                current_test_path = os.path.join(current_path, 'test')
                output_path_test = os.path.join(output_path, 'test')
                off_to_mat_converter.convert_off_to_mat(current_test_path, output_path_test)
print("End of Program")                
