# 3D-Dataset-Preprocessing-Pipeline
A data preprocessing pipeline designed to prepare 3D geometric data for training machine learning models, more specifically, 3D Convolutional Neural Networks (CNNs).

## 3D Dataset Creation

A data preprocessing pipeline for creating 3D data targets to train ML models, particularly CNN. These 3D data were derived from the 2D EMNIST dataset and other 3D datasets. 
The process of target creation included:

**1) Simple Targets:** These were created by extruding 2D EMNIST images in one of the three dimensions (x, y, or z) with random lengths and starting points. This produced simple 3D image objects from the 2D data.

**2) Complex Targets:** The dataset of simple targets was divided into two halves. The first half was combined with (added to) the second half in different dimensions, resulting in more complex 3D structures of EMNIST Mashup dataset. The pixel values were adjusted such that any non-zero value was set to 255, creating a binarylike complex object. The addition of  different simple targets led to some complex targets with discontinuities in them (two separated volumes in a space).

**3) 3D ShapeNets:** In addition to the EMNIST-derived data, 3D ShapeNets were also incorporated. This is a dataset representing 3D volumetric shapes like chairs, toilets, beds, and airplanes. This dataset was created by researchers from Princeton University in Princeton's ModelNet Object Database.



## 3) 3D ShapeNets:
A data preprocessing pipeline for converting 3D mesh datasets (specifically ModelNet10/40) into voxelized HDF5 datasets suitable for 3D Deep Learning models (like 3D CNNs).

### Features
**Format Conversion:** Batch converts .off (Object File Format) meshes to .mat (MATLAB) files.

**Voxelization:** Converts 3D meshes into occupancy grids (voxels) with normalization and scaling.

**Dataset Packaging:** Splits data into Train/Test/Validation sets and saves everything into a single, compressed HDF5 file.

**Configurable:** Supports custom voxel resolutions (default: 32x32x32).

### To install:

`pip install -r requirements.txt`

### Usage
There are two main steps to generating the dataset.

**Step 1:** Convert OFF to MAT
Run the directory navigation script to convert raw mesh files into MATLAB format. This script filters specific categories by default.

`python 3DShapeNets_OFF_directory_navigate_and_convert.py`

**Input:** Directory containing .off files (e.g., ModelNet40/ModelNet40_off).

**Output:** Directory for .mat files.

**Step 2:** Voxelize and Package
Run the preparation script to convert the .mat files into the final HDF5 dataset.

`python prepare_voxel_data.py`

**Input:** Directory containing the .mat files generated in Step 1.

**Prompts:** You will be asked for the target voxel size (default: 32) and output filename.

**Output:** An .hdf5 file containing train_mat, test_mat, val_mat and their labels.


### Output Format
The resulting HDF5 file contains:

train_mat: Voxelized training data.

test_mat: Voxelized testing data.

val_mat: Voxelized validation data (split from test set).




# Credit
Princeton University ModelNet

Available:
https://modelnet.cs.princeton.edu/

Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao
3D ShapeNets: A Deep Representation for Volumetric Shapes
Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)
Oral Presentation ·  3D Deep Learning Project Webpage
