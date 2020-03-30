# -*- coding: utf-8 -*-
"""Best final submission file"""
import csv
import numpy as np
from process_data import *
from proj1_helpers import *
from implementations import *

PRI_jet_num_index = 22

def create_submission(optimal_degrees, optimal_lambdas, DATA_TRAIN_PATH, DATA_TEST_PATH, SUBMISSION_PATH):
    
    print("Step 1/3: Computing optimal weights using the train dataset...")
    # Compute the optimal w for each subset using the full train dataset.
    w_subsets = compute_ws_on_train_dataset(DATA_TRAIN_PATH, optimal_degrees, optimal_lambdas)
    
    print("Step 2/3: Computing the final prediction using the test dataset...")
    # Compute the prediction using the test dataset and the best w.
    y_final_prediction, ids_test = compute_y_pred_on_test_dataset(DATA_TEST_PATH, w_subsets, optimal_degrees)
    
    print("Step 3/3: Create the submission file")
    # Create the submission file
    create_csv_submission(ids_test, y_final_prediction, SUBMISSION_PATH)
    
    print("Task completed !")
    return 

def compute_ws_on_train_dataset(DATA_TRAIN_PATH, optimal_degrees, optimal_lambdas):
    """ Given the train dataset, compute the best_ws for each subset (split using PRI_jet_num)"""
    
    step = 1
    
    
    print("Step 1-{s}/5: Train data loading...".format(s=step))
    step += 1
    # Load the train data from csv file.
    y, tX, _ = load_csv_data(DATA_TRAIN_PATH) 
    
    # Compute the differents values of PRI_jet_num
    jet_values = [int(x) for x in np.unique(tX[:, PRI_jet_num_index])]
    
    
    # Compute the indices list for each subsets 
    subsets_indices_array = indices_split_dataset_jet_num(tX, PRI_jet_num_index, jet_values)
    
    # Compute the subsets and delete useless columns
    x_subsets, all_deleted_columns_indices = clean_useless_columns_jet(tX, subsets_indices_array)

    ws = []
    
    # Iterate over the subsets
    for i in jet_values:
        
        print("Step 1-{s}/5: Compute w for subset {i}".format(i=i, s=step))
        step += 1
        
        # Compute the y vector associated to a data subset
        y_subset = y[subsets_indices_array[i]]
        
        # Apply the data processing on a subset
        x_subset_processed = full_process_data(x_subsets[i], optimal_degrees[i],
                                               DATA_TRAIN_PATH, all_deleted_columns_indices[i], False)
        
        # Compute the weight vector for a subset using ridge regression
        w_i, _ = ridge_regression(y_subset, x_subset_processed, optimal_lambdas[i])
        
        # Add the w associated to current subset into the array of w.
        ws.append(w_i) 
    return ws

def compute_y_pred_on_test_dataset(DATA_TEST_PATH, ws, optimal_degrees):
    step = 1
    
    print("Step 1-{s}/5: Test data loading...".format(s=step))
    step += 1
    # Load the test data from csv file.
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    
    # Compute the differents values of PRI_jet_num
    jet_values = [int(x) for x in np.unique(tX_test[:, PRI_jet_num_index])]
    
    # Compute the indices list for each subsets
    subsets_indices_array = indices_split_dataset_jet_num(tX_test, PRI_jet_num_index, jet_values)
    
    # Compute th substes and delete useless columns
    x_test_subsets, all_deleted_columns_indices = clean_useless_columns_jet(tX_test, subsets_indices_array)
    
    # Initalize the final prediction vector
    y_final_prediction = np.zeros(tX_test.shape[0]);
    
    # Iterate over the subsets
    for i in jet_values:
        
        print("Step 1-{s}/5: Compute prediction for subset {i}".format(i=i, s=step))
        step += 1
        # Apply the data processing on a subset
        x_test_processed = full_process_data(x_test_subsets[i], optimal_degrees[i], 
                                             DATA_TEST_PATH, all_deleted_columns_indices[i], False)
        # Compute the prediction using processed data and associated w.
        y_pred = predict_labels(ws[i], x_test_processed)
        
        # Add the prediction into the final array using the right indices
        y_final_prediction[subsets_indices_array[i]] = y_pred
      
    # Return the final prediction and associated ids for submission
    return y_final_prediction, ids_test    

def indices_split_dataset_jet_num(x, PRI_jet_num_index, jet_values):
    """Return indices to split the full dataset based on the PRI_jet_num feature"""
    return [np.where(x[:, PRI_jet_num_index] == i) for i in jet_values]

def clean_useless_columns_jet(x, subsets_indices_array):
    """Delete the useless columns of all ."""
    result = []
    all_useless_indices = [] 
    
    # Itereate over the subsets
    for jet_val, indice_set in enumerate(subsets_indices_array):
        
        # Create a new dataset with the subset indices
        dataset = x[indice_set]
        useless_indices = []
        
        # Iterate over subset column
        for i in range(dataset.shape[1]):
            
            # Count the number of values for a column
            count = len(np.unique(dataset[:, i]))
            
            # If a column has only one values, add its index to the useless indices
            if count == 1:
                useless_indices.append(i)
        
        all_useless_indices.append(useless_indices)
        # Delete the useless columns
        dataset = np.delete(dataset, useless_indices, axis=1)
        # Add the subset to the resulting array of datasets
        result.append(dataset) 
            
    return result, all_useless_indices
    