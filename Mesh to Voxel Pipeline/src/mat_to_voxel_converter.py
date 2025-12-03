# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:54:33 2024

@author: ken
"""


"""
This program converts A SINGLE 3D mesh data from .mat file into 
a voxel grid, allowing 3D models to be represented as a grid of 
voxels. It normalizes the vertices of the mesh, voxelizes the 
3D model, and resizes it to a specified voxel size. The final 
voxel grid is optionally padded to fit a target shape.


### Note to Users:
# 
# I encountered various compatibility issues when trying to use different 
# versions of the required libraries. Specifically, some lines of the code, 
# such as "voxels = voxelized.matrix.astype(np.float32)", may not work with
# certain versions of the Trimesh library. Additionally, some versions of 
# Trimesh are not compatible with specific versions of TensorFlow and NumPy, 
# leading to further complications.
#
# After significant troubleshooting, I found a combination of library versions
# that works without any errors. To ensure that this code runs correctly, 
# I recommend using the following versions:
#     - Numpy: 1.19.5
#     - Tensorflow: 2.6.2
#     - Trimesh: 3.23.5
#
# Using these versions should help avoid the issues I encountered and allow 
# the code to run smoothly.
"""


import numpy as np
import scipy.io
import trimesh
import skimage.measure


"""
This function loads a .mat file containing 3D mesh data and 
extracts the vertices and faces from it. Returns these 
vertices and faces as NumPy arrays for further processing.
"""
def load_mat_file(file_path):
    mat_contents = scipy.io.loadmat(file_path)
    vertices = mat_contents['vertices']
    faces = mat_contents['faces']
    return vertices, faces



"""
This function normalizes the vertex positions to fit within 
a unit cube (range of [0,1]) and then scales them to fit 
within a voxel grid of the specified size. It makes sure 
that the 3D model is properly scaled before voxelization.
"""
def normalize_vertices(vertices, voxel_size):
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    normalized_vertices = (vertices - min_bounds) / (max_bounds - min_bounds)
    scaled_vertices = normalized_vertices * (voxel_size - 1)
    print(f"scaled_vertices: {scaled_vertices}")
    return scaled_vertices



"""
This function converts the 3D mesh, defined by the vertices 
and faces, into a voxel grid using a specified voxel size.
It downsamples or resizes the voxel grid to match the desired
resolution if necessary.
"""
def mesh_to_voxel(vertices, faces, voxel_size):
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Use the trimesh.voxelize function to create a voxel grid
    voxelized = mesh.voxelized(pitch=1.0)
    
    #trimesh.exchange.binvox.voxelize_mesh(mesh)
    
    #voxels = voxelized.grid()
    #voxels = voxels.astype(np.float32)  
    # Convert voxel grid to 3D numpy array
    voxels = voxelized.matrix.astype(np.float32)    

    # Resize to 32x32x32 if needed
    #print(voxels.shape)
    
    downsample_factors = (
    max(1, voxels.shape[0] // voxel_size),
    max(1, voxels.shape[1] // voxel_size),
    max(1, voxels.shape[2] // voxel_size)
    )
    
    if voxels.shape != (voxel_size, voxel_size, voxel_size):
        #voxels = skimage.measure.block_reduce(voxels, (voxels.shape[0] // voxel_size, voxels.shape[1] // voxel_size, voxels.shape[2] // voxel_size), np.max)
        voxels = skimage.measure.block_reduce(voxels, downsample_factors, np.max)
    
    return voxels

# File path to the .mat file
#file_path = 'ModelNet10/Mat/chair/train/chair_0001.mat'

# Load the vertices and faces
#vertices, faces = load_mat_file(file_path)

"""
This function is the main function that handles user input for 
the intended voxel size and defaults to 32 if the input is 
invalid. It normalizes vertices, voxelizes the mesh, and then 
pads the voxel grid to ensure it matches the target shape.
"""
def convert_mat_to_voxel(vertices, faces, voxel_size):
    # Normalize and scale vertices to fit within the voxel grid     
    scaled_vertices = normalize_vertices(vertices, voxel_size)
    # Convert the mesh to voxels
    if np.isnan(scaled_vertices).any() or np.isnan(faces).any() or scaled_vertices.size == 0 or faces.size == 0:
        print("\nnan found\n")
        return None
    
    voxels = mesh_to_voxel(scaled_vertices, faces, voxel_size)    
    
    # Define the target shape
    target_shape = (voxel_size, voxel_size, voxel_size)
    
    # Calculate padding for each dimension
    pad_width = [
        (0, target_shape[i] - voxels.shape[i]) for i in range(3)
    ]
    
    # Apply padding to the voxel array
    padded_voxels = np.pad(voxels, pad_width, mode='constant', constant_values=0)
    
    #print("Original shape:", voxels.shape)
    #print("Padded shape:", padded_voxels.shape)
    
    return padded_voxels
    
#    # Now 'voxels' is a numpy array with shape (32, 32, 32)
#    #from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    x, y, z = np.indices((32, 32, 32))
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
#    filename = 'AAAA_chair001b' + '.png'
#    #path = os.path.join(output_path, filename)
#    plt.savefig(filename)