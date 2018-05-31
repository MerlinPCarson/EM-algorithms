 ##########################
##  Merlin Carson         ##
##  CS446, Spring 2018    ##
##  k-means EM-algorithm  ##
 ##########################
import sys
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.cm import rainbow
style.use('ggplot')
import numpy as np
from random import randint
#for Gaussian
from scipy.stats import multivariate_normal
from collections import Counter
from math import log
import math



K = 2
MAX_TIMES = 100
THRESHOLD = 0.001
TRAINING_SIZE = 1500
MIN_BUBBLE_EXAMPLES = TRAINING_SIZE * 0.1 # If a bubble has less data, reset algorithm
GMM_RANDOM_INIT = False#True    # Randomly init GMM or init using k-means clusters

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


def init_clusters():
    # create list for data in each cluster
    clusters = {}
    for cluster in range(K):
        clusters[cluster] = list()

    return clusters


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

    # use k-means clusters for inititial parameters
    if not GMM_RANDOM_INIT:
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

    #means = np.array([[0,0], [1,1]])
    #covs = np.array([[[0,0],[0,0]],[[1,1],[1,1]]]) 
    #priors = np.array([0.5,0.5])
    
    else:
        # randomly initialize parameters
        for cluster in range(K):
            means.append(np.random.randn(2))
            A = np.random.randn(2,2)
            covs.append(np.dot(A,A.transpose()))
            priors.append(float(1)/K)

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

def mv_normal(example, mean, cov):
    normalization = 1/(2*math.pi*np.linalg.det(cov)**(1/2))
    sub =  np.array(example - mean)
    trans = np.transpose(np.array(sub)[None])
    inverse = np.linalg.inv(cov)
    term2 = np.array(np.dot(inverse,sub))
    term1 = np.reshape(trans,2)
    exp = np.matmul(term1,term2)
    value = math.exp(-1/2*exp)
    return normalization * value

def responsibility(example, cluster, means, covs, priors):
    normalization = 0
    #likelihood = mv_normal(example,means[cluster], covs[cluster])
    #likelihood = multivariate_normal(means[cluster],covs[cluster]).pdf(example)
    likelihood = multivariate_normal.pdf(example,means[cluster],covs[cluster])
    for k in range(len(means)):
        normalization += priors[k]*multivariate_normal(means[k], covs[k]).pdf(example)

    return priors[cluster]*likelihood/normalization


def GMMexpectation(data, clusters, means, covs, priors):
    responsibilities = list()

    # calculate responsiblities
    for example in data:
        gammaK = list()
        for cluster in clusters:
            gammaK.append(responsibility(example,cluster,means,covs,priors))

        # create array of responsibilities 
        responsibilities.append(gammaK)

    for example in range(len(data)):
        idx = np.argmax(responsibilities[example])
        clusters[idx].append(data[example])

    return responsibilities


def GMMmaximization(data, responsibilities):
    
   
    # calculate N of k
    Ns = list()
    for cluster in range(K):
        Nk  = 0
        for example in range(len(data)):
            Nk += responsibilities[example][cluster]
        Ns.append(Nk)

    # calculate new means
    means = list()
    for cluster in range(K):
        mean = np.zeros([2])
        for example in range(len(data)):
            mean += responsibilities[example][cluster] * data[example]
        means.append(mean/Ns[cluster])

    # calculate new covariance matricies
    covs = list()
    for cluster in range(K):
        cov = covariance(data, responsibilities, means[cluster])
        covs.append(cov/Ns[cluster])

    # calculate new priors
    priors = list()
    for cluster in range(K):
        priors.append(Ns[cluster]/len(data))


    return means, covs, priors


# calculate log likelihood of the algorithm
def loglikelihood(data, means, covs, priors):
    #clusters = list()
    llikelihood = 0
    for example in data:
        posterior = 0
        for cluster in range(K):
            posterior += priors[cluster]*multivariate_normal(means[cluster],covs[cluster]).pdf(example)
        llikelihood += log(posterior)
    
    print llikelihood
    return llikelihood


# determines if any clusters are too small 
def check_bubble_sizes(bubbles, minSize):

    for bubble in bubbles:
        if len(bubbles[bubble]) < minSize:
            print (len(bubbles[bubble]), ' in a cluster, resetting!')
            return True

    return False


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
    clusters = init_clusters()

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

    print 'Converged after {} iterations.'.format(times)
    
    # display clusters and centroids
    #plot(centroids, clusters)
    
    # clear out centroids, not needed for GMM
    centroids = {}

    ##############################
    #Start Gaussian Mixture Model
    ##############################

    reset = True

    while reset:
        reset = False

        # inizialize parameters
        means, covs, priors = GMMinitialize(clusters)
    
        times = 0
        llikelihood = 0
        converged = False
        # run EM steps MAX_TIMES or until the centroids are finished moving
        while times < MAX_TIMES and not converged and not reset:

            # initialize data structures for cluster data
            bubbles = init_clusters()

            # caluculate log likely hood to classify
            responsibilities = GMMexpectation(data, clusters, means, covs, priors)

            # update parameters
            means, covs, priors = GMMmaximization(data, responsibilities)

            # determine log likelihood
            prev_llikelihood = llikelihood
            llikelihood = loglikelihood(data, means, covs, priors)
            if abs(llikelihood - prev_llikelihood) < THRESHOLD:
                converged = True

            reset = check_bubble_sizes(bubbles, MIN_BUBBLE_EXAMPLES)

            times += 1

    # display parameters
    print 'Means-\n\n', 
    for mean in means:
        print mean, '\n'
    print 'Covariance Matricies-\n '
    for cov in covs:
        for entry in cov:
            print entry
        print
    print 'Priors-\n'
    for prior in priors:
        print prior, '\n'
    
    # display clusters and centroids
    plot(centroids, bubbles)

