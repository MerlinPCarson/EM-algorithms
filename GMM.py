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
#for Gaussian
from scipy.stats import multivariate_normal


K = 2
MAX_TIMES = 100
THRESHOLD = 0.001
TRAINING_SIZE = 100

# plot all data
#def plot(data):
#    plt.scatter(data[:,0], data[:,1], s=10)
#    plt.show()


#
# K-Means functions
#############################


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


#
# Gaussian Mixure Models functions
#############################

def GMMinitialize(clusters):

    means  = list()
    covs   = list()
    priors = list()
    totalSize = 0

    #calculate total number(N) of data points
    for cluster in clusters:
        totalSize += len(clusters[cluster])

    #calculate the means of each cluster
    for cluster in clusters:
        means.append(np.sum(clusters[cluster], axis = 0)/ len(clusters[cluster]))
        cov = np.zeros([2,2])
        for example in clusters[cluster]:
            term2 = np.array(example-means[cluster])[None]
            term1 = np.transpose(term2)
            cov += np.matmul(term1, term2)
        covs.append(cov/len(clusters[cluster]))
            
    #calculate the covariance matricies

    #calculate the priors for each cluster
    for cluster in clusters:
        #print len(clusters[cluster])
        priors.append(float(len(clusters[cluster]))/totalSize)

    #print means
    #print covs
    #print priors

    return means, covs, priors

def gaussian(example, mean, cov, prior):
    pass


def covariance(data, responsibilites, mean):
    cov = np.zeros([2,2])
    for example in data:
        term2 = np.array(example-mean)[None]
        term1 = np.transpose(term2)
        cov += np.matmul(term1, term2)

    return cov


def responsibility(example, cluster, means, covs, priors):
    normalization = 0
    likelihood = multivariate_normal(means[cluster],covs[cluster]).pdf(example)
    for k in range(len(means)):
        normalization += multivariate_normal(means[k], covs[k]).pdf(example)

    return priors[cluster]*likelihood/normalization


def GMMexpectation(data, means, covs, priors):
    responsibilities = list()

    for example in data:
        gammaK = list()
        for cluster in clusters:
            gammaK.append(responsibility(example,cluster,means,covs,priors))

        # create array of responsibilities 
        responsibilities.append(gammaK)

    return responsibilities


def GMMmaximization(data, numK, responsibilities):
    
   
    # calculate N of k
    Ns = list()
    for cluster in range(numK):
        Nk  = 0
        for example in range(len(data)):
            Nk += responsibilities[example][cluster]
        Ns.append(Nk)

    # calculate new means
    means = list()
    for cluster in range(numK):
        mean = np.zeros([2])
        for example in range(len(data)):
            mean += responsibilities[example][cluster] * data[example]
        means.append(mean/Ns[cluster])

    # calculate new covariance matricies
    covs = list()
    for cluster in range(numK):
        cov = covariance(data, responsibilities, mean[cluster])
        covs.append(cov/Ns[cluster])

    # calculate new priors
    priors = list()
    for cluster in range(numK):
        priors.append(Ns[cluster]/numK)


    return means, covs, priors


def loglikelihood(data, means, covs, priors):
    clusters = list()

    return clusters


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
    converged = False
    
    # run EM steps MAX_TIMES or until the centroids are finished moving
    while times < MAX_TIMES and not converged:

        
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
        converged = threshold(centroids, prev_centroids, THRESHOLD)

    print 'number of iterations: ', times
    
    ##############################
    #Start Gaussian Mixture Model
    ##############################

    # inizialize parameters
    means, covs, priors = GMMinitialize(clusters)
    
    times = 0
    converged = False
    # run EM steps MAX_TIMES or until the centroids are finished moving
    while times < MAX_TIMES and not converged:

        # initialize data structures for cluster data
        clusters = {}
        for cluster in range(K):
            clusters[cluster] = list()

        # caluculate log likely hood to classify
        responsibilities = GMMexpectation(data, means, covs, priors)

        # update parameters
        means, covs, priors = GMMmaximization(data, K, responsibilities)

        # determine cluster for each datum
        clusters = loglikelihood(data, means, covs, priors)
    
        times += 1
    
    # display clusters and centroids
    plot(centroids, clusters)

