#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    # raise Exception("Not implemented yet")
    dictionary = {}
    words = x.split()
    for word in words:
        if word in dictionary:
            dictionary[word] = dictionary[word] + 1
        else:
            dictionary[word] = 1
    return dictionary
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    numIters refers to a variable you need to declare. It is not passed in.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    # raise Exception("Not implemented yet")
    numIters = 100
    stepSize = .1

    def predictor(x):
        dp = dotProduct(weights,x)
        if(dp > 0):
            return 1
        else:
            return -1

    def gradient(weights,feature,y):
        if len(feature) == 0:
            return 0
        value = dotProduct(weights,feature)*y
        if value < 1:
            ret = {}
            for k,v in feature.items():
                ret[k] = v*y*-1
            return ret
        else:
            return 0

    for i in range(numIters):
        for x,y in trainExamples:
            feature = featureExtractor(x)
            ret = gradient(weights,feature,y)
            if ret != 0:
                increment(weights,-1*stepSize,ret) 
        #evaluatePredictor(trainExamples, predictor)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        # raise Exception("Not implemented yet")
        phi = {}
        numKeys = random.randint(1,len(weights))
        for i in range(numKeys):
            key = random.choice(weights.keys())
            phi[key] = random.randint(1,20)
        score = dotProduct(weights,phi)
        if score > 0:
            y = 1
        else:
            y = -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3f: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        # raise Exception("Not implemented yet")
        feature = {}
        x = x.replace(" ", "")
        for i in range(0,len(x)-n+1):
            word = x[i:i+n]
            if word in feature:
                feature[word] = feature[word] + 1
            else:
                feature[word] = 1
        return feature
        # END_YOUR_CODE
    return extract

############################################################
# Problem 3h: extra credit features

def extractExtraCreditFeatures(x):
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    # raise Exception("Not implemented yet")
    # Assuming data points are unique
    def distance(point1,point2):
        sum = 0
        for k,v in point1.items():
            dif = (point1[k] + point2[k])**2
            sum = sum+dif
        return math.sqrt(sum)

    def classify(centers,point):
        minDistance = sys.maxint
        minCenter = 0
        for i in range(len(centers)):
            center = centers[i]
            dist = distance(center,point)
            if dist < minDistance:
                minDistance = dist
                minCenter = i
        return minCenter

    def calculateCenters(centers, clusters):
       for i in range(len(centers)):
            newCenter = {}
            data = clusters[i] #list of maps
            for point in data: #a map
                for pointKey,pointValue in point.items():
                    if pointKey in newCenter:
                        newCenter[pointKey] = newCenter[pointKey] + point[pointKey]
                    else:
                        newCenter[pointKey] = point[pointKey]
            numDims = len(newCenter)
            for centerKey,centerValue in newCenter.items():
                newCenter[centerKey] = newCenter[centerKey]/numDims
            centers[i] = newCenter
    
    def kMeansLoss(centers,assignments,examples):
        loss = 0
        for i in range(len(examples)):
            example = examples[i]
            center = centers[assignments[i]]
            loss = loss + distance(example,center)**2
        return loss

    # assignments: example number -> cluster number
    assignments = range(len(examples))
    # clusters: cluster number -> cluster center point
    centers = random.sample(examples,K)
    # for every iteration
    for i in range(maxIters):
        # clusters: cluster number -> list of examples
        clusters = {}
        # print("centers: ",centers)
        for p in range(K):
            clusters[p] = []    
        # go through all the examples and assign to the correct center
        for j in range(len(examples)):
            example = examples[j]
            numCenter = classify(centers,example)
            assignments[j] = numCenter
            clusters[numCenter].append(examples[j])
        # recalculate the centers
        # print("clusters",clusters)
        calculateCenters(centers,clusters)
    totalCost = kMeansLoss(centers,assignments,examples)
    print(totalCost,": totalCost")
    return centers, assignments, totalCost
    # END_YOUR_CODE
