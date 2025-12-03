# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 01:25:15 2024

@author: ken
"""


"""
This script converts all files in a single directory containing
3D geometry data from OFF files into a format compatible with 
MATLAB (MAT files), and save them to a specified directory. The 
main function, convert_off_to_mat, handles the conversion process, 
iterating over OFF files in the input directory and saving the 
converted MAT files to the output directory.

The main function, convert_off_to_mat is called from another 
program which provides parameters of the input directory containing
the OFF files and the output directory to save the MAT files to, 
this program converts the OFF files in the input directory to MAT
files and stores them in the output directory.
"""


import os
import numpy as np
from scipy.io import savemat
import glob



"""
This function reads an OFF file, extracting vertices and faces 
data. It checks for a valid OFF header, reads the number of 
vertices and faces, and stores them as numpy arrays.
"""
def read_off(file):
    with open(file, 'r') as f:
        head_line = f.readline().strip()
        if head_line.startswith("OFF"):
            off_label = head_line[3:]  # Extract 'OFF'
            off_label = off_label.strip()
            if off_label != "":
                n_verts, n_faces, _ = map(int, off_label.split())        
            else:            
                n_verts, n_faces, _ = map(int, f.readline().strip().split())
            verts = []
            for _ in range(n_verts):
                verts.append(list(map(float, f.readline().strip().split())))
            faces = []
            for _ in range(n_faces):
                faces.append(list(map(int, f.readline().strip().split()[1:])))
        else:
            return None, None
    return np.array(verts), np.array(faces)



"""
This function saves the vertices and faces data into a MAT file 
using scipy.io.savemat, making it compatible with MATLAB.
"""
def save_mat(file, verts, faces):
    savemat(file, {'vertices': verts, 'faces': faces})



"""
This functions converts all OFF files in a specified directory to MAT files,
saving them in a designated output directory. It also ensures 
the output directory exists.
"""
def convert_off_to_mat(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # find all files with the .off extension in the input_dir
    off_files = glob.glob(os.path.join(input_dir, '*.off'))
    
    for off_file in off_files:
        verts, faces = read_off(off_file)
        if verts is not None and faces is not None:
            mat_file = os.path.join(output_dir, os.path.splitext(os.path.basename(off_file))[0] + '.mat')
            save_mat(mat_file, verts, faces)
            #print(f'Converted {off_file} to {mat_file}')

                
                
