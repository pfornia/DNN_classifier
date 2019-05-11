from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers


def kmeans(data, k, verbose = False):
   

    #Follow psuedocode from pg 167 of Alpaydin
    #initialize k centroids. 
    centroids = pd.DataFrame(index = range(k), columns = data.columns)

    for col in centroids:
        for i in range(centroids.shape[0]):
            centroids[col][i] = random.random() 

    assignments = [0]*data.shape[0]
    for iter in range(100):
        if verbose: print("iteration: " + str(iter))

        #Make new assignments
        old_assignments = copy.deepcopy(assignments)
        for i, obs in data.iterrows(): assignments[i], _ = helpers.nearest_obs(obs, centroids) 
        num_obs_change = np.sum([old_assignments[i] != a for i,a in enumerate(assignments)])
        if verbose: print(str(num_obs_change) + " of " + str(data.shape[0]) + " observations changed clusters.")
        if num_obs_change == 0:
            return assignments, centroids

        #Update centroids
        for c in range(centroids.shape[0]):
            clust_data = data[[a == c for a in assignments]]
            for col in clust_data:
                if len(clust_data) > 0:
                    centroids[col][c] = np.mean(clust_data[col])

    if verbose: print("warning, kmeans did not converge after " + str(iter + 1) + " iterations")
    return assignments, centroids



