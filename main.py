from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import pickle
import helpers
import ann
import rbf
import sys

#Input Data Set and parameters...

#If inputs not provide, then prompt user.
if len(sys.argv) == 3:
    selection = int(sys.argv[1])
    filepath = sys.argv[2]
else:
    print("Please select classification model type...")
    print("1. Neural Net")
    print("2. Radial Basis Function Network")
    selection = int(input("Selection: "))
    filepath = input("Enter file path of data: ")

if selection == 1: 
    hidden_layers = int(input("Please select number of hidden layers: "))
    hidden_nodes = []
    for layer in range(hidden_layers):
        hidden_nodes += [int(input("How many nodes in layer " + str(layer) + ": "))]
    
    print("Running NN, with hidden layers of size:")
    print(hidden_nodes)

if selection == 2:
    rbf_k = int(input("Please select number of centroids/kernels/hidden nodes for RBF: "))

#Read in cleaned dataset.
clean_data = pd.read_pickle(filepath)

#Get list of features with "target" in the name.
targets = []
for col in clean_data.columns.values:
    if 'target' in col: targets += [col]

#Set seed for train/test splits.
random.seed(3)

#Report the accuracy of simply predicting the most frequent target class.
print("Naive Baseline Accuracy:")
print(np.max([sum(clean_data[t]) for t in targets])/clean_data.shape[0])

folds = helpers.k_folds_split(clean_data, k=5)

performance = [0 for x in folds]

#Train and predict algorithm on each fold
for i,f in enumerate(folds):

    print("Training/testing on fold: " + str(i))
    
    if selection == 1:
        model = ann.nn_train(f[0], 
            hidden_layers = hidden_layers, 
            hidden_nodes = hidden_nodes, 
            targets = targets)
        estimates = ann.nn_predict(model, f[1], targets = targets)
 
    elif selection == 2:
        model = rbf.rbf_train(f[0], targets = targets, sigma = 1, k = rbf_k)
        estimates = rbf.rbf_predict(model, f[1])
 
    #Print output of first fold
    if i == 0:
        print("Classifications for first fold:")
        print(estimates)   
    actuals = f[1][targets].values
    print("Accuracy of this fold:")
    performance[i] = ann.nn_accuracy(actuals, estimates)
    print(performance[i])

#Report the average performance across all 5 folds.
print("Average performance accross folds:")
print(np.mean(performance))


