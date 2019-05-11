from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd

def dotprod(v1, v2):
    """
    Perform a dot product of two vectors
    """
    v1 = [float(v) for v in v1]
    v2 = [float(v) for v in v2]

    f = 0.0
    for j,v in enumerate(v1):
        f += v*v2[j]
        
    return f

def replace_missing_mode(dataset, miss_char = '?'):
    """
    Replace all missing values (e.g., '?' or blank) with the most frequently occuring value from a column
    """
    for col in dataset:

        freqs = dataset[col].value_counts()
        mode = np.argmax(freqs)

        for i,d in enumerate(dataset[col]):
            if d == miss_char:
                dataset[col][i] = mode
                
    return(dataset)
    

def one_hot_encode(dataset, exclude = ['target'], real_quartiles = False):
    """
    Perform one-hot encoding, to make multi-level categorical variables into multiple, binary variables
    """
    for col in dataset:
        if col in exclude:
            next
        elif type(dataset[col][0]) == str:
            print(col)
            col_levels = dataset[col].drop_duplicates()
                
            #col_levels = dataset[col].value_counts()
            for level in col_levels:
                dataset[col + "_" + str(level)] = 0
        
            for i,d in enumerate(dataset[col]): dataset.loc[i, col + "_" + str(d)] = 1

            dataset = dataset.drop([col], axis = 1)
        elif type(dataset[col][0]) in [float, np.float64] and real_quartiles:
            print(col)
            quartiles = dataset[col].quantile([0.25,0.5,0.75]).values
            print(quartiles)
            dataset[col + "_q1"] = 0         
            dataset[col + "_q2"] = 0        
            dataset[col + "_q3"] = 0        
            dataset[col + "_q4"] = 0 
            for i,d in enumerate(dataset[col]): 
                if d < quartiles[0]:               
                    dataset.loc[i, col + "_q1"] = 1         
                elif d < quartiles[1]:               
                    dataset.loc[i, col + "_q2"] = 1         
                elif d < quartiles[2]:               
                    dataset.loc[i, col + "_q3"] = 1         
                else:               
                    dataset.loc[i, col + "_q4"] = 1
            
            dataset = dataset.drop([col], axis = 1) 
         
    return dataset

def normalize(dataset, exclude = ['target']):
    """
    Perform normalization of real valued features to range from -1 to 1.
    """
    for col in dataset:
        if col in exclude:
            next
        elif type(dataset[col][0]) in [float, np.float64]:
            min_val = np.min(dataset[col])
            max_val = np.max(dataset[col])
            middle = (max_val + min_val)/2
            range = max_val - min_val 
            new_vals = [(x - middle)/(0.5*range) for x in dataset[col]]
            dataset[col] = new_vals

    return dataset

def confusion(actuals, predictions):
    """
    Produce the confusion matrix for actuals and predicted values from a two-class classification problem.
    """
    if(len(actuals) != len(predictions)): 
        print("Must evaluate same number of actual and predicted values")
        return
        
    tot = len(actuals)
    act_pos = np.sum(actuals)
    act_neg = tot - act_pos
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for i,a in enumerate(actuals):
        p = predictions[i]
        if p == 0:
            if a == 0:
                tn += 1
            else:
                fn += 1
        else:
            if a == 0:
                fp += 1
            else:
                tp += 1            
    
    print("Confusion Matrix:")
    print("\t\t\tActual Values")
    print("\t\t\tPos\tNeg")
    print("Predicted\tPos\t" + str(tp) + "\t" + str(fp))
    print("\t\tNeg\t" + str(fn) + "\t" + str(tn))
    print()
    print("Accuracy:\t\t\t" + str((tp+tn)/tot))
    print()
    
def train_test_split(dataset, train_ratio = 0.667):
    """
    Split a pandas dataframe into a train and test set with specified of train to total.
    """
    np.random.seed(1)
    dataset_shuf = dataset.iloc[np.random.permutation(len(dataset))]
    dataset_shuf = dataset_shuf.reset_index(drop=True)
    
    train_count = round(len(dataset) * train_ratio)
    
    dataset_train = dataset_shuf[:train_count].reset_index(drop=True)
    dataset_test = dataset_shuf[train_count:].reset_index(drop=True)
    
    return dataset_train, dataset_test

def k_folds_split(dataset, k = 5):
    """
    split the dataset into k folds, each containing 1/k of the total observations.

    inputs:
    dataset: the original dataset to be split
    k: the number of desired folds.

    returns:
    A list of folds.
    Each fold is a 2-tuple containing 1) the train dataframe, and 2) the test dataframe
    """
    
    np.random.seed(1)
    dataset_shuf = dataset.iloc[np.random.permutation(len(dataset))]
    dataset_shuf = dataset_shuf.reset_index(drop=True)

    n = dataset_shuf.shape[0]

    all_folds = []
    for f in range(k):
        split_beg = int(np.floor(n*f/k))
        split_end = int(np.floor(n*(f+1)/k))
        data_train1 = dataset_shuf[:split_beg]
        data_train2 = dataset_shuf[split_end:]
        data_train = data_train1.append(data_train2).reset_index(drop = True)
        data_test = dataset_shuf[split_beg:split_end].reset_index(drop = True)

        all_folds += [(data_train, data_test)]

    return(all_folds)

def k_folds_stratified(dataset, k=5, strat_var = 'target'):
    """
    split the dataset into k folds, stratified by a target variable.

    inputs:
        dataset: the original dataset to be split
        k: the number of desired folds.
        strat_var: the name of the target variable on which to stratify    
 
    returns:
    A list of folds.
    Each fold is a 2-tuple containing 1) the train dataframe, and 2) the test dataframe
    """

    all_folds = None
    for c in list(set(dataset[strat_var])): 
        data_class = dataset[dataset[strat_var] == c]
        folds_class = k_folds_split(data_class, k)
        if all_folds == None: 
            all_folds = copy.deepcopy(folds_class)
        else:
            for i,f in enumerate(folds_class):
                all_folds[i] = (all_folds[i][0].append(f[0]), all_folds[i][1].append(f[1])) 

    for i,f in enumerate(all_folds):
        all_folds[i] = (f[0].iloc[np.random.permutation(len(f[0]))].reset_index(drop=True), f[1].iloc[np.random.permutation(len(f[1]))].reset_index(drop=True))

    return all_folds


def distance(v1, v2):
    these_distances = [(v - v2[i])**2 for i,v in enumerate(v1)]
    this_dist = np.sqrt(np.sum(these_distances))
    return this_dist

def nearest_obs(this_obs, candidates):
    near_obs = -1
    near_obs_distance = np.inf

    for i, row in candidates.iterrows():
        this_dist = distance(this_obs, row)
 
        #min distance
        if this_dist < near_obs_distance:
            near_obs_distance = this_dist
            near_obs = i 

    return near_obs, near_obs_distance
