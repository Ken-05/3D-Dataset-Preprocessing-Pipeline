# coding: utf-8

# Import Libraries for use 
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


# ## First Convert 2D Images to 3D
# #### Extrude in x,y, or z dimension randomly at a random length
# ###### First 1/3 of data extruded in x dimension
# ###### Second 1/3 of data extruded in y dimension
# ###### Third 1/3 of data extruded in z dimension


default_train_size = 27000
default_test_size = 3000
default_box_size = 28

# Ask the user to input numbers
try:
    train_size = int(input(f"Enter length of train samples to create, Default train length is '{default_train_size}': "))   
except ValueError:
    print(f"Invalid input. Using default train length size value of '{default_train_size}'.")
    train_size = default_train_size
try:
    test_size = int(input(f"Enter length of test samples to create, Default test length is '{default_test_size}': "))   
except ValueError:
    print(f"Invalid input. Using default test length size value of '{default_test_size}'.")
    test_size = default_test_size
try:
    box_size = int(input(f"Enter box size of voxels to create, Default box size is '{default_box_size}': "))
except ValueError:
    print(f"Invalid input. Using default box size value of '{default_box_size}'.")
    box_size = default_box_size


(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Create a total of 30000 simple targets
x_train3d = np.zeros((train_size,box_size,box_size,box_size)) 
x_test3d = np.zeros((test_size,box_size,box_size,box_size))  

# Create a total of 30000 simple targets
#x_train3d = np.zeros((27000,28,28,28)) 
#x_test3d = np.zeros((3000,28,28,28))   

#train_size = x_train3d.shape[0]
#test_size = x_test3d.shape[0]

trainsize_third = int(train_size/3)
testsize_third = int(test_size/3)
twice_trainsize_third = int(2*trainsize_third)
twice_testsize_third = int(2*testsize_third)
trainsize_half = int(train_size/2)
testsize_half = int(test_size/2)


# Prepare training samples
for i in range(train_size):         
    # Extract the current 2D image and extrude into 3D
    train_2d_image = x_train[i]            # get the 2D image
    
    # randomly determine extrusion length
    rand_start = random.randint(0, box_size-3)
    rand_length = random.randint(3, 8)   # using random length from 3 to 8
    rand_end = min((rand_start+rand_length),box_size)
    
    # For a third of the total train samples, extrude the 2D image 
    # a random length in z direction for the 3D object
    if i<trainsize_third:
        for j in range(rand_start,rand_end):            
            x_train3d[i, j, :, :] = train_2d_image     
    # For another third of the total train samples, extrude the 2D 
    # image a random length in x direction for the 3D object
    elif i<twice_trainsize_third:
        for j in range(rand_start,rand_end):            
            x_train3d[i, :, j, :] = train_2d_image     
    # For another third of the total train samples, extrude the 2D 
    # image a random length in y direction for the 3D object
    else:
        for j in range(rand_start,rand_end):            
            x_train3d[i, :, :, j] = train_2d_image            
            
            
# Prepare testing samples
for i in range(test_size):
    # Extract the current 2D image and extrude into 3D
    test_2d_image = x_test[i]            # get the 2D image
    
    rand_start = random.randint(0, box_size-3)
    rand_length = random.randint(3, 8)
    rand_end = min((rand_start+rand_length),box_size)
        
    if i<testsize_third:        
        for j in range(rand_start,rand_end):            
            x_test3d[i, j, :, :] = test_2d_image    
    elif i<2*twice_testsize_third:
        for j in range(rand_start,rand_end):            
            x_test3d[i, :, j, :] = test_2d_image     
    else:
        for j in range(rand_start,rand_end):            
            x_test3d[i, :, :, j] = test_2d_image                         



#print(x_train3d.shape)
#print(x_test3d.shape)


# Display Some 2D Images

# Create array to use to display random 2D images
show_array = [(0,0),(trainsize_third,testsize_third),(twice_trainsize_third,twice_testsize_third)]
for i,j in show_array:
    fig, ((ax1, ax2)) = plt.subplots(1,2, figsize=(9,9))
    a1 = ax1.imshow(x_train[i])
    a2 = ax2.imshow(x_test[j])
    fig.tight_layout()
    plt.show()


# Display Some 3D Object

for i,j in show_array:
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(121, projection='3d')  # First subplot for x_train
    ax2 = fig.add_subplot(122, projection='3d')  # Second subplot for x_test

    ax1.voxels(x_train3d[i], edgecolor='k')  # Plot the 3D volume in ax1
    ax2.voxels(x_test3d[j], edgecolor='k')   # Plot the 3D volume in ax2

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax1.set_title('x_train')
    ax2.set_title('x_test')

    # Show the plot
    plt.tight_layout()
    plt.show()


# ## Next Convert 3D Objects to More Complex 3D Objects
# ##### Divide total data size into two (for combination later)
# ##### Add first half to second half to create 2 different extrusion dimensions to give more complex 3D data in all dimensions
# ###### Pixel value of new complex object remains 0 if it is 0
# ###### Pixel value of new complex object becomes to 255 if it is greater than 0

# In[6]:


x_train3dcomplex = np.zeros((trainsize_half,box_size,box_size,box_size))
x_test3dcomplex = np.zeros((testsize_half,box_size,box_size,box_size))

# Combine first half of training samples to second half
for i in range(trainsize_half):
    temp1 = x_train3d[i]
    temp2 = x_train3d[i+trainsize_half]
    tempcomplex = temp1+temp2
    tempcomplex[tempcomplex >= 1] = 255
    x_train3dcomplex[i] = tempcomplex

# Combine first half of test samples to second half    
for i in range(testsize_half):
    temp3 = x_test3d[i]
    temp4 = x_test3d[i+testsize_half]    
    tempcomplex2 = temp3+temp4
    tempcomplex2[tempcomplex2 >= 1] = 255     
    x_test3dcomplex[i] = tempcomplex2    
    
#print(x_train3dcomplex.shape)
#print(x_test3dcomplex.shape)


# Display Some Complex 3D Object

# Create array to use to display random voxel plots
complex_show_array = [(0,0),(1,1),(2,2),(3,3),(4,4),
                      (trainsize_third,testsize_third),
                      (trainsize_third+1,testsize_third+1),
                      (trainsize_third+2,testsize_third+2),
                      (trainsize_third+3,testsize_third+3),
                      (trainsize_third+4,testsize_third+4), 
                      (trainsize_half-1,testsize_half-1)]

for i,j in complex_show_array:
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(121, projection='3d')  # First subplot for x_train
    ax2 = fig.add_subplot(122, projection='3d')  # Second subplot for x_test

    ax1.voxels(x_train3dcomplex[i], edgecolor='k')  # Plot the 3D volume in ax1
    ax2.voxels(x_test3dcomplex[j], edgecolor='k')   # Plot the 3D volume in ax2

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax1.set_title('x_train')
    ax2.set_title('x_test')

    # Show the plot
    plt.tight_layout()
    plt.show()


# Store complex 3D Objects into a file for future use

# Add an extra dimension to the train and test data (L,x,x,x,1)
x_train3dcomplex = np.expand_dims(x_train3dcomplex, -1).astype("float32") / 255
x_test3dcomplex = np.expand_dims(x_test3dcomplex, -1).astype("float32") / 255


import pickle
with open('complex3DMashupTargets.pkl', 'wb') as file:
    pickle.dump({'x_train3dcomplex': x_train3dcomplex, 'x_test3dcomplex': x_test3dcomplex}, file)
    
'''
# To Load the data from the pickle file for use
import pickle
with open('complex3DMashupTargets.pkl', 'rb') as file:
    new_targets = pickle.load(file)
x_train3dcomplex = new_targets['x_train3dcomplex']
x_test3dcomplex = new_targets['x_test3dcomplex']
print(x_train3dcomplex.shape)
print(x_test3dcomplex.shape)
'''

print("End of Program")

