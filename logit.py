from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers


def logit_train(dataset, target = 'target', eta = 0.1, max_iter = 1000):
    """
    Train the logistic regression. See Mod 09 video 2, around 9:00

    Input: 
        dataset: A training dataset
        target: The target feature name
        eta: A learning rate
        max_iter: Maximum number of iterations to perform gradient descent

    Output:
        A vector of weights that make up the logistic regression model. 
        The first element is the 'bias', and is to be applied to a feature
        of ones, not included in dataset.
    """
    
    ys = dataset[target]
    xs = dataset.drop([target], axis = 1).values
    xs = [[1] + list(x) for x in xs] 

    n,m = len(xs), len(xs[0])

    weights = [random.random()*0.02 - 0.01 for x in range(m)]
    
    #while True:
    for iteration in range(max_iter):
        deltas = [0 for x in weights]
        for t in range(n):
            o = np.dot(weights, xs[t])
            y = 1/(1+np.exp(-1*o))
            for j,d in enumerate(deltas):
                deltas[j] = d + (ys[t] - y)*xs[t][j]
        epsilon = eta*np.sqrt(np.sum([d**2 for d in deltas])/len(deltas))
        weights = [(weights[i] + eta*d) for i,d in enumerate(deltas)] 

        if iteration%10 == 0: 
            print("Iteration " + str(iteration) + ":  \tError = " + str(epsilon))

        if epsilon < 0.01: break

    return weights
    
    
def logit_predict(logit_model, dataset, target = 'target'):
    """
    Form predictions using a logistic regression model.

    Input:
        logit_model: A model output from logit_train function.
        dataset: A test dataset.
        target: The name of the target feature.
    """

    #xs = dataset.drop([target], axis = 1).values
    xs = copy.deepcopy(dataset.values)
    xs = [[1] + list(x) for x in xs]

    ypred = [1/(1+np.exp(-1*np.dot(x, logit_model))) for x in xs]
        
    return ypred


def logit_train_multiclass(dataset, targets = ['target'], eta = 0.1):
    """
    Wrapper function for logit_train to handle multi-class classification.
    For each target class, the function runs logit_train to train a
    one-vs-all (OVA) classification.

    Input:
        dataset: Training dataset
        target: The name of the target feature.
        eta: Learning rate

    Output:
        A dictionary of logit models -- one for each target class. 
    """

    #Dictionary of one-vs-all logit models with key of class label.
    ova_outputs = {}
    
    multi_target = dataset[targets]

    classes = multi_target.drop_duplicates()

    for t in targets:
        newdata = dataset.drop(targets, axis=1)
        newdata[t] = multi_target[t]
        ova_outputs[t] = logit_train(newdata, eta = eta, target = t)
    
    return ova_outputs

def logit_predict_multiclass(logit_model_multi, dataset):
    """
    Predict the target class from the multi-class dictionary of OVA logit
    models.

    Inputs:
        logit_model_multi: Dictionary of logit models, output of logit_train_multiclass
        dataset: Test dataset
        target: The name of the target feature.

    Output:
        Classification predictions for the test set observations.
    """
   
    xs = copy.deepcopy(dataset)    
 
    #for t in logit_model_multi: xs = xs.drop(t)

    predictions = []

    for t in logit_model_multi:
        this_model = logit_model_multi[t]
        predictions += [logit_predict(this_model, dataset, target = t)] 
    
    #get highest scored class for each observation
    class_pred = []
    for i in range(len(predictions[0])):
        probs = [x[i] for x in predictions]
        preds = [0 for p in probs]
        preds[np.argmax(probs)] = 1
        class_pred += [preds]

    return class_pred
