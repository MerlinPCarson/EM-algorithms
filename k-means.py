 ##########################
##  Merlin Carson         ##
##  CS446, Spring 2018    ##
##  k-means EM-algorithm  ##
 ##########################

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.cm import rainbow
style.use('ggplot')
import numpy as np
from random import randint


K = 2
MAX_TIMES = 100
THRESHOLD = 0.001
TRAINING_SIZE = 1500

# plot all data
#def plot(data):
#    plt.scatter(data[:,0], data[:,1], s=10)
#    plt.show()

# plot clusters by color
def plot(centroids, clusters):
    print ('Plotting...')
    # draw centroids
    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], marker="o", color="k", linewidths=5, s=5)
    
    colors = iter(rainbow(np.linspace(0, 1, len(clusters))))
    # draw cluster
    for cluster in clusters:

        # pick a color from the rainbow list
        cluster_color = next(colors)

        # plot each example within a cluster
        for example in clusters[cluster]:
            plt.scatter(example[0], example[1], marker = "x", color = cluster_color,  linewidths=5, s=5)   
    
    # display graph
    plt.legend(fontsize="small")
    plt.show()


# randomly intialize centroids
def initialize(data, centroids):
    tmpData = list(data)

    for centroid in range(K):
        selection = randint(0,K)
        centroids[centroid] = tmpData[selection]
        tmpData = np.delete(tmpData,selection, axis=0)
    

# determines centroid each datum is closests to    
def expectation(data, centroids, clusters):
    
    for example in data:
        distances = [np.linalg.norm(example-centroids[centroid]) for centroid in centroids]
        cluster = distances.index(min(distances))
        clusters[cluster].append(example)

    return centroids


# calculates new centroids
def maximization(data, centroids, clusters):

    for cluster in clusters:
        centroids[cluster] = np.average(clusters[cluster], axis=0)


# determines when algorithm has optimized enough
def threshold(centroids, prev_centroids, THRESHOLD):
    # determine how far the centroids moved
    for centroid in range(len(centroids)):
        change = np.sum((centroids[centroid]-prev_centroids[centroid])/prev_centroids[centroid])

        # if one moved more than the THRESHOLD, keep running algorithm
        if change > THRESHOLD:
            return False
    
    return True


# predicts a new datum
def predict(example, centroids):
    distances = [np.linalg.norm(example-centroids[centroid]) for centroid in centroids]
    return distances.index(min(distances))


########################################
# MAIN PROGRAM
########################################

# load data from file
data = np.genfromtxt('GMM_dataset.txt')
data = data[:TRAINING_SIZE]

# User prompts
K = input("How many clusters would you like: ")
iterations = input("How many iterations of k-means would you like: ")


# how many times the algorithm should run
for _iter in range(iterations):
    # dictionary for the centroids
    centroids = {}
    # create list for data in each centroid
    clusters = {}
    for centroid in range(K):
        clusters[centroid] = list()

    initialize(data, centroids)
    
    times = 0
    finished = False
    
    # run EM stepts MAX_TIMES or until the centroids are finished moving
    while times < MAX_TIMES and not finished:

        
        # initialize data structures
        prev_centroids = dict(centroids)
        clusters = {}
        for centroid in range(K):
            clusters[centroid] = list()

        # determine the clusters
        expectation(data, centroids, clusters)
        
        # recalculate the centroids
        maximization(data, centroids, clusters)
        
        times += 1

        # check change in distance threshold to determine when to stop
        finished = threshold(centroids, prev_centroids, THRESHOLD)

    print 'number of iterations: ', times
    
    # display clusters and centroids
    plot(centroids, clusters)

