from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers

def y_predict_layer(thetas, xs):
    """
    Create estimates for one layer of nodes from a list of thetas and the node values 
        from the previous layer. Can be used for multiple observations at once.

    Inputs:
        thetas: List of lists of logistic regression parameters or "thetas" for use in 
            an ANN. Each element of the outer list is aligned to one node in the layer
            being predicted. Each element of this inner list is a weight for a single 
            edge in the network.
        xs: list of lists containing the n-by-m matrix of values from the previous
            layer (i.e., actual inputs, or hidden layer values). Each element of the 
            outer list is aligned to one observation. Each element of the inner list 
            corresponds to a node.

    Output: 
        A list of lists of node values (i.e., hidden layer values, or outputs). Each 
        of the lists corresponds to an observation, and each element of the list 
        corresponds to a node in the output layer. These are probabalistic, already 
        having been transformed by the logistic activation function.
    """
    zs = []
    for xi in xs:
        zs += [[sum([xi[j]*t for j,t in enumerate(thetai)]) for thetai in thetas]]
     
    ys = []
    for zi in zs:
        ys += [[1/(1+np.exp(-1*z)) for z in zi]]
    
    return(ys)

def calculate_error_multiclass(actuals, predicted):
    """
    Calculate the accuracy rate for a classification problem with 
    one-hot-encoded target values.

    Inputs:
        actuals: an n-by-k list of lists containing n observations of
            actual target values with k possible classes, one-hot-
            encoded into k fields.

        predicted: an n-by-k list of lists of predicted (same form
            as actuals).
    """
    actuals_unwrapped = []
    predicted_unwrapped = []
    for i,a in enumerate(actuals):
        actuals_unwrapped += list(a)
        predicted_unwrapped += list(predicted[i])
    errors = [y*np.log(predicted_unwrapped[i]) + (1 - y)*np.log(1-predicted_unwrapped[i]) for i, y in enumerate(actuals_unwrapped)]
    return(-1/len(errors)*sum(errors))

def forward_backprop(xs, ys, theta_h_prev, theta_o_prev, alpha):
    """
    Perform the feedforward and backpropagation step for one observation. 
    Takes one observation of data and the old model, and returns a new model.

    Inputs:
        xs: A list of x values for one observation (image data)
        ys: The list of binaries containing the true classification for one observation.
        theta_h_prev: List of lists containing existing theta weights for the hidden layer.
        theta_o_prev: List of lists containing existing theta weights for the output layer.
        alpha: The current learning rate for the ANN algorithms. 

    Output:
        The model, represented as a tuple of Thetas. The first is corresponds to the hidden 
        layer, and the second to the output layer. Each of these is a list of lists. Each 
        element of the outer list corresponds to the theta vector that trains one node in 
        the hidden (or output) layer. Each element of this inner list corresponds to the 
        weight on one edge of the ANN, coming from the input (or hidden) layer.

    """

    #Feed-forward step
    #  track the node values
    y_hid_hats = []
    prev_layer = xs
    for thetas_layer in theta_h_prev:
        y_hid_layer = y_predict_layer(thetas_layer, [prev_layer])[0]
        y_hid_hats += [y_hid_layer]
        prev_layer = [1] + y_hid_layer

    y_out_hats = y_predict_layer(theta_o_prev, [prev_layer])[0]

    #Backpropagation step
    deltaOs = [yh*(1-yh)*(ys[i] - yh) for i, yh in enumerate(y_out_hats)]
    deltaHs = []

    above_deltas = deltaOs
    above_thetas = theta_o_prev
    for step in range(len(y_hid_hats)):
        layer = len(y_hid_hats) - step - 1
      
        above_deltas = [yh*(1-yh)*np.sum([above_thetas[j][i+1]*d
            for j,d in enumerate(above_deltas)]) for i, yh in enumerate(y_hid_hats[layer])]
        
        #Append delta layer to **beginning** of list (since working backwards)
        deltaHs = [above_deltas] + deltaHs 
        above_thetas = theta_h_prev[layer]
        
    theta_h_new = []
    lower_layer = xs
    for layer in range(len(theta_h_prev)):    
        theta_h_new += [[[t + alpha*deltaHs[layer][j]*(lower_layer)[i] 
            for i,t in enumerate(theta)] for j,theta in enumerate(theta_h_prev[layer])]]
        lower_layer = [1] + y_hid_hats[layer]

    theta_o_new = [[t + alpha*deltaOs[j]*(lower_layer)[i] 
        for i,t in enumerate(theta)] for j,theta in enumerate(theta_o_prev)]

    return(theta_h_new, theta_o_new)

def nn_train(dataset, hidden_layers = 1, hidden_nodes = [10], targets = ['target'], verbose = True):
    """
    Train an Artificial Neural Network (ANN) on a dataset with multi-class labels.

    Inputs:
        dataset: Dataset to train
        hidden_layers: number of hidden layers
        hidden_nodes: List of length hidden_layers, describing
            the number of hidden nodes in each layer.
        targets: List of the names of the one-hot-encoded feature
            variables.
        verbose: If true, print additional information as the model trains.

    Output: 
        The NN model, represented as a 3-dimensional matrix of edge weights.
            The outer most dimension represents the layer,
            The second dimension represents the node in the next "downstream" layer.
            The third dimension represents the index of the "upstream" node.

    """
    alpha = 1
    epsilon = 1e-4
    iteration = 0
    
    ys = dataset[targets].values
    xs = copy.deepcopy(dataset)
    xs['intercept'] = 1
    xs = xs.drop(targets, axis = 1).values

    n, x_nodes, y_nodes = xs.shape[0], xs.shape[1], ys.shape[1]
    
    hidden_thetas = []
    lower_node_count = x_nodes    

    for layer in range(hidden_layers): 
        hidden_thetas += [[[random.random()-0.5 for x in range(lower_node_count)] 
            for x in range(hidden_nodes[layer])]]
        lower_node_count = hidden_nodes[layer] + 1
        
    output_thetas = [[random.random()-0.5 for x in range(lower_node_count)] for x in range(y_nodes)]

    current_error, previous_error = 20.0, 10.0
   
    while abs(current_error - previous_error) > epsilon:
        for obs in range(n):
            hidden_thetas, output_thetas = forward_backprop(xs[obs], ys[obs,:], hidden_thetas, output_thetas, alpha)
 
        prev_layer = xs
        for thetas_layer in hidden_thetas:      
            y_hid_hats = y_predict_layer(thetas_layer, prev_layer) 
            prev_layer = [[1] + yh for yh in y_hid_hats]       
    
        y_out_hats = y_predict_layer(output_thetas, prev_layer) 
        
        iteration += 1
        previous_error = current_error
        current_error = calculate_error_multiclass(ys, y_out_hats)
        
        if current_error > previous_error: 
            alpha /= 10.0
            if verbose: print("Increased Error; Shrinking Alpha to " + str(alpha))

        if verbose and iteration%10 == 0:  print("Iteration " + str(iteration) + ":  \tError = " + str(current_error))
    
    return((hidden_thetas, output_thetas))

def nn_predict( model, test_data, targets, labeled=False):
    """   
    Create predicted classifications of a test dataset based on a NN model.

    Input:      
        model: output from nn_train function, format described above.
        test_data: Test data set on which to produce predictions
        targets: List of the names of the one-hot-encoded feature
            variables. 

    Output:
        an n-by-k list of lists containing n observations of
            predicted target values with k possible classes, one-hot-
            encoded into k fields.
    """
    if labeled:
        xs, ys = [d[0:-1] for d in test_data], [d[-1] for d in test_data]
        predictions = apply_model(model, xs, labeled=False)
        return([[(ya, predictions[i][j][0]) for j,ya in enumerate(yactuals)] for i,yactuals in enumerate(ys)])
        
    else:
        hidden_thetas = model[0]
        output_thetas = model[1]

        #xs_full = [[1] + x for x in test_data]
        xs = copy.deepcopy(test_data)
        xs['intercept'] = 1
        xs = xs.drop(targets, axis = 1).values
       
        #y_hid_hats = y_predict_layer(hidden_thetas, xs) 
        #y_hid_full = [[1] + yh for yh in y_hid_hats]
        #y_out_hats = y_predict_layer(output_thetas, y_hid_full) 
        
        prev_layer = xs
        for thetas_layer in hidden_thetas:      
            y_hid_hats = y_predict_layer(thetas_layer, prev_layer) 
            prev_layer = [[1] + yh for yh in y_hid_hats]       

        y_out_hats = y_predict_layer(output_thetas, prev_layer) 

        ypreds = [[0 for yh in yhats] for yhats in y_out_hats]

        for i, yhats in enumerate(y_out_hats): 
            ypreds[i][np.argmax(list(yhats))] = 1

        return(ypreds)

def nn_accuracy(actuals, predictions):
    """
    Calculated accuracy of one-hot-encoded predictions against actuals.

    Inputs:
        actuals: an n-by-k list of lists containing n observations of
            actual target values with k possible classes, one-hot-
            encoded into k fields.
        predictions: an n-by-k list of lists containing n observations of
            predicted target values with k possible classes, one-hot-
            encoded into k fields.

    Output:
        Accuracy of predictions.
    """
    tot = actuals.shape[0]
    correct = 0.0
    for i, row in enumerate(actuals):
        if list(row) == predictions[i]: correct += 1

    return(correct/tot)

 
