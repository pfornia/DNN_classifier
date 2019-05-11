from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers
import kmeans
import logit

def rbf(x, center, sigma):
    """
    The Radial basis function; similar to Gaussian pdf.
    """
    return np.exp(-1/(2*(sigma**2))*(helpers.distance(x, center)**2))

def rbf_train(dataset, targets, sigma, k):
    """
    Train the RBF network algorithm.

    Inputs:
        dataset: Training dataset.
        targets: List of one-hot-encoded target variable.
        sigma: spread of gaussian kernels.
        k: number of centroids/kernels.

    Output:
        RBF Model: Sigma, k centroids, and logistic regression weights.

    """
    ys = dataset[targets]
    xs = copy.deepcopy(dataset)
    xs = xs.drop(targets, axis = 1)

    #get centroids
    print("Finding Centroids...")
    _, centroids = kmeans.kmeans(xs, k)

    #build df of rbf values
    print("Finding RBF values...")
    hidden_vals = pd.DataFrame(index = xs.index, columns = centroids.index)

    for i in range(hidden_vals.shape[0]):
        for j in range(hidden_vals.shape[1]):
            hidden_vals.loc[i,j] = rbf(xs.loc[i,:], centroids.loc[j,:], sigma)    

    hidden_vals[targets] = ys

    #gradient descent to train output layer weights
    #  (equivalent to logistic regression on rbf nodes)
    print("Training output layer weights...")
    weights = logit.logit_train_multiclass(hidden_vals, targets, eta = 0.01)

    return {'sigma': sigma,
            'centroids': centroids,
            'weights': weights}

def rbf_predict(model, dataset):
    """
    Form predictions from the RBF model.

    Inputs:
        model: RBF network model from rbf_train, described above.
        dataset: Test dataset on which to form predictions.

    Outputs:
        One-hot encoded prediction values for each observations in dataset.
    """
    centroids = model['centroids']
    
    xs = copy.deepcopy(dataset)
    for t in model['weights']: xs = xs.drop(t, axis = 1)   
 
    #build df of rbf values
    hidden_vals = pd.DataFrame(index = xs.index, columns = centroids.index)

    for i in range(hidden_vals.shape[0]):
        for j in range(hidden_vals.shape[1]):
            hidden_vals.loc[i,j] = rbf(xs.loc[i,:], centroids.loc[j,:], model['sigma'])    
    
    predictions = logit.logit_predict_multiclass(model['weights'], hidden_vals)

    return predictions
