# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


def initial_process_data(x, verbose=False):
    if verbose: 
        print("Replace outliers (-999 and mean +- 3 std dev)")
        
    x = replace_outliers(x) # Replace the outliers and the NaN values
    
    if verbose: 
        print(str(x.shape))
        print("Standardize columns") 
        
    x = standardize_cols(x, verbose) # Standardize the columns 
    
    if verbose:
        print(str(x.shape))
        print("Add first column")
        
    
    x = build_model_data(x) # Add the column of 1.
    
    if verbose:
        print(str(x.shape))
        
    return x


def full_process_data(x, degree, data_path, useless_indices, verbose=False):
    if verbose:
        print("Replace outliers (-999 and mean +- 3 std dev)")
        
    x = replace_outliers(x)  # Replace the outliers and the NaN values
    
    if verbose:
        print(str(x.shape))
        print("Add angle features (cos, sin)")
        
    x = add_angle_features(x, data_path, useless_indices) # Add sin and cos for features that are angles
    
    if verbose: 
        print(str(x.shape))
        print("Standardize columns")
        
    x = standardize_cols(x, verbose) # Standardize the columns 
    
    if verbose:
        print(str(x.shape))
        print("Build polynomial features of degree {d}".format(d=degree))
        
    x = build_poly(x, degree, verbose) # Build polynomial features
    
    if verbose:
        print(str(x.shape))
        print("Standardize columns")
        
    x = standardize_cols(x, verbose) # Standardize the columns 
    
    if verbose: 
        print(str(x.shape))
        print("Add first column")
        
    x = build_model_data(x) # Add the column of 1.
    
    if verbose:
        print(str(x.shape))
    
        
    return x

def show_columns_distribution(x):
    """ Plow the dataset columns using poxplot"""
    for i in range(x.shape[1]):
        plt.boxplot(x[:, i])
        plt.title("Column {i}".format(i=i))
        plt.show()
    return

def get_final_angle_indices(data_path, useless_indices):
    """Compute the angle indices for a subset of the dataset after some useless features has been removed"""
    
    angle_features_indices = get_initial_angle_indices(data_path) # Get initial indices of angle features
    dataset_final_angle_indices = []
    
    for angle_index in angle_features_indices: # For each initial angle feature 
        smaller_count = 0
        is_deleted = False
        for del_index in useless_indices:  # For each deleted features
            if del_index == angle_index: # If angle feature has been deleted, ignore it
                is_deleted = True
            elif del_index < angle_index: # If a deleted feature is before an angle one, update it's index
                smaller_count += 1
        if not(is_deleted):
            dataset_final_angle_indices.append(angle_index - smaller_count) # Add it to the result if not deleted
    return dataset_final_angle_indices
            

def get_initial_angle_indices(data_path):
    """ Get the indices of angle features from the csv file"""
    # Load the header from the dataset csv file.
    header = np.genfromtxt(data_path, delimiter=",", skip_header=0, dtype=str, max_rows=1)[2:]
    angle_features_indices = []
    for i, h in enumerate(header): # For each features
        if 'phi' in h: # If a features contains phi (e.g. is an angle)
            angle_features_indices.append(i) # Add the feature to the final result
    return angle_features_indices
    
    
def add_angle_features(x, data_path, useless_indices):     
    """ Add angle features to the dataset"""
    # Get the angle features indices
    angle_features_indices = get_final_angle_indices(data_path, useless_indices)
        
    for angle_i in angle_features_indices: # For each angle feature
        sin_col = np.sin(x[:, angle_i])
        cos_col = np.cos(x[:, angle_i])
        x = np.c_[x, sin_col] # Add the sin of the angle
        x = np.c_[x, cos_col] # Add the cos of the angle
    return x
    
    
def standardize(x):
    """Standardize a column a of dataset."""
    #Compute the mean and standard deviation
    mean_x = np.mean(x)
    std_x = np.std(x)
    #Normalize x
    x = (x - mean_x) / std_x
    return x, mean_x, std_x

def standardize_cols(x, verbose=True):
    """Standardize the columns of a dataset """
    num_features = x.shape[1]
    result = np.zeros(x.shape[0])
    #Iterate on the columns of the dataset
    for i in range(num_features):
        if verbose and i % 10 == 0:
            print("Standardize col: {n}/{last}".format(n=i,last=num_features-1)) 
        #Standardize the column i
        std, _, _ = standardize(x[:,i])
        #Add the column to the result
        result = np.c_[result, std]
    #Return the result with the first zeros column
    return result[:,1:]

def replace_outliers(x):
    """Replace the outlines and the -999 values by the median."""
    x_null = np.where(x==-999, np.nan, x)
    median = np.nanmedian(x_null, axis=0)
    for i in range(x.shape[1]):
        x[:, i][x[:, i] == -999] = median[i]
    #Once we have replaces the -999, we can compute the mean and std and replace outliers
    for i in range(x.shape[1]):
        mean = np.mean(x[:, i])
        std = np.std(x[:, i])
        #Replace values that are bigger than mean + 3std or smaller than mean - 3std
        x[:, i][x[:, i] > mean + 3*std] = median[i]
        x[:, i][x[:, i] < mean - 3*std] = median[i]
    return x


def build_model_data(x):
    """Form (y,tX) to get regression data in matrix form."""
    return np.c_[np.ones(len(x)), x]



def build_poly(x, degree, verbose=True):
    """Build different degrees of existing features"""
    result = np.zeros((len(x), 1)) # Initalize result matrix
    for deg in range(1, degree + 1): # Iterage over degrees
        if verbose:
            print("Build polynomial degree: {n}/{last}".format(n=deg,last=degree))  
        result = np.c_[result, np.power(x, deg)] # Add the new degree to the result matrix

    return result[:,1:] # Return all the columns except the useless first one



        