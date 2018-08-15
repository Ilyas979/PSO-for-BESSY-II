# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:49:46 2017

@author: J.Li, M.Ruprecht, HZB
"""
from __future__ import division
from functools import partial
import numpy as np
import random
import sys
import copy

def testFunction(X):
    #:::: ZDT1 ::::#
    n = X.shape[0]
    G  = 1 + (9*(np.sum(X) - X[0])/(n-1)) # or G  = 1 + 9*(np.sum(X[2:n]))/(n-1)
    F1 = X[0]
    F2 = G*(1 - np.sqrt(np.divide(X[0],G)))
    F = np.array([F1, F2])

    #:::: ZDT3 ::::#
    #n = X.shape[0]
    #G  = 1 + 9*(np.sum(X[2:n]))/(n-1)
    #F2 = G*(1 - np.sqrt(np.divide(X[0],G)) - np.divide(X[0],G)*math.sin(10*math.pi*X[0]))
    #F1 = X[0]
    #F = np.array([F1, F2])

    return F

def lhs(swarmsize,Dim):
    samples = swarmsize
    N = Dim
    rng = np.random

    segsize = 1.0 / samples

    rdrange = rng.rand(samples,N) * segsize
    rdrange += np.atleast_2d(np.linspace(0.0,1.0,samples,endpoint = False)).T

    x = np.zeros_like(rdrange)

    for j in range(N):
        order = rng.permutation(range(samples))
        x[:,j] = rdrange[order,j]
    return x

def Dominates(X,Y):
    b = False
    if all(X<=Y) and any(X<Y):
        b = True
    return b
def Mutate(x, pm, mu, lb, ub):
    x = list(x)
    nVar = len(x)
    nMu = np.ceil(mu*nVar)
    j = random.sample(range(nVar), int(nMu))
   # print 'pm: ', pm                       
    sigma =  pm * (ub - lb)         
   # print 'sigma1: ', sigma
    if len(sigma)>1:                #can make it shorter and more intuitive
        sigma = [sigma[i] for i in j]
   # print 'sigma2: ', sigma
    y = copy.deepcopy(x)
    rerror = np.random.rand(len(j))*sigma
    for v in range(int(nMu)):
        y[j[v]] = x[j[v]]+rerror[v]
     #   if y[j[v]] != x[j[v]]: print 'not eq'

        if y[j[v]] < lb[j[v]]:
            y[j[v]] = lb[j[v]]
        if y[j[v]] > ub[j[v]]:
            y[j[v]] = ub[j[v]]
    
    return np.array(y)


def nondomSolutions2(X, F):
    X = np.array(X)
    F = np.array(F)
    nF = len(F[:,0]) #Number of functions, it is not
    nD = np.zeros((nF,1))
    index = []
    for i in xrange(0, nF):
        nD[i][0] = 0
        for j in xrange(0,nF):
            if j != i:
                if (all(F[j,:] <= F[i,:]) and any(F[j,:] < F[i,:]) ):
                    nD[i][0] = 1 #that means ith solution is bad
        if nD[i][0] == 0:
            index.append(i)
    repX = X[index,:]
    repF = F[index,:]
    it = 0    #why do we call it like this?
    nMax = len(repX[:,0])
    #print 'repX.shape: ', repX.shape, 'repX[0,:].shape: ', repX[0,:].shape
    while (it < nMax):
        uIdx = sum((nF == (0 == (repX - repX[it,:])).sum(1)).astype(int)) #what do you mean here?
        if uIdx > 1:
            repX = np.delete(repX, it, axis = 0)
            repF = np.delete(repF, it, axis = 0)
            nMax = len(repX[:,0])
            it = 0
        else:
            it = it + 1
            nMax = len(repX[:,0])
    return repX, repF
def nondomSolutions(X, F):
    X = np.array(X)
    F = np.array(F)##this will be removed when we wrap it in class
    nSolutions = len(F[:,0]) #Number of solutions
    index = []
    for i in range(0, nSolutions):
        nD = 0
        for j in range(0, nSolutions):
            if (all(F[j,:] <= F[i,:]) and any(F[j,:] < F[i,:]) and i != j): #J dominates I
                nD = 1 #that means ith solution is bad, beacuse there is a better one - jth
                break
        if nD == 0:
            index.append(i)
    repX = X[index,:]
    repF = F[index,:]
    repX, idx = np.unique(repX, axis = 0, return_index = True)
    repF = repF[idx,:]
    return repX, repF

def inverse_permutation_crowding(p):
    #This function computes an "inverse permutation"
    return np.array([p.index(l) for l in range(len(p))])
def crowding_sorting(archiveX, archiveF):
    nondomN = len(archiveF)
    Nobj = len(archiveF[0,:])
    crowdDist = np.zeros((nondomN, ))
    for i in range(Nobj):
        indexes = np.argsort(archiveF[:,i])
        archivetestF = archiveF[indexes, i]
        crowdDist = crowdDist[indexes] #(*)
        for j in range(1, nondomN-1):
            crowdDist[j] = crowdDist[j] + (archivetestF[j+1]-archivetestF[j-1])/(archivetestF[-1]-archivetestF[0])
        crowdDist[0] = np.inf
        crowdDist[-1] = np.inf
        #at the end of each loop apply an inverse permutation to crowdDist,
        #because here (*) we should apply a permutation to initailly ordered array
        invert_indexes = inverse_permutation_crowding(list(indexes))
        crowdDist = crowdDist[invert_indexes]

    indx = np.argsort(crowdDist)[::-1]
    archiveX = archiveX[indx]
    archiveF = archiveF[indx]
    return archiveX, archiveF

def removeparticles(archiveX,archiveF,N):
	for i in range(int(N)):
		archiveX=np.delete(archiveX,-1,axis=0)
		archiveF=np.delete(archiveF,-1,axis=0)
	return archiveX,archiveF
def SelectLeader(archiveX,archiveF):
	Nx = np.floor(len(archiveX)*0.1)
	h = random.randint(0, Nx)
	return h

def _obj_wrapper(func, x):
    return func(x)


def mopsocd(func,lb,ub,Nobj,swarmsize=200, debug = False, maxit=60,nRep=100,w=0.4,wdamp=0.99,c1=2.8,c2=1.4,alpha=0.1,mu=0.05,bait=[],processes=1,seed=None):

    """
    costFunction:
    lb: lower boundary of variables to be optimized
    ub: upper boundary of variables to be optimized
    swarmsize: the number of particles
    maxit: maximum iterations
    nRep: the number of particles saved in the repository
    w: final inetia factor
    wdamp: intial inertia facor
    c1: cognitive facotr
    c2: social learning factor
    alpha: the portion of choosing leader in the respository
    mu: mutation rate
    bait: should be a list(consisting of list),to speed up the convergency if some good setting already found
    processes: for multiprocesses
    seed: seed for the random generators
    """
##################initialization

    # Setting the seed, so that reproducible runs are possible
    random.seed(seed)
    np.random.seed(seed)

    costFunction = partial(_obj_wrapper, func)

    #print random.getstate()

    vmax = (ub-lb)

    D=len(ub)
    #X = np.random.rand(swarmsize, D)  # particle positions
    X = lhs(swarmsize,D)
    V = np.zeros_like(X)  # particle velocities
    F= np.zeros((swarmsize,Nobj))  # current particle function values
    PXbest = np.zeros_like(X)  # best particle positions
    PFbest = np.ones(swarmsize)*np.inf  # best particle function values

    # Initialize the particle's position
    X = lb + X*(ub - lb)

    Vhigh = np.abs(ub - lb)
    Vlow = -Vhigh
    V = Vlow + lhs(swarmsize, D)*(Vhigh - Vlow)
    if bait !=[]:
        idxx = np.random.randint(0,high=swarmsize,size=len(bait))
        if debug: print idxx,swarmsize
        for idx,elem in zip(idxx,bait):

            X[idx,:]=np.array(elem)

            if debug: print X[idx,:], 'bait has been used!'

    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        if debug: print 'multiprocessing works'
        F= np.array(mp_pool.map(costFunction, X))
    else:
        for i in range(swarmsize):
            F[i]= costFunction(X[i, :])
#            if debug: print F[i]
    if debug: print 'the 0th iteration finished'

    #Initialize personal best by initial positions
    PXbest=X.copy()
    PFbest=F.copy()

#########################initialization is finished

    nondomX, nondomF = nondomSolutions(X,F)

    archiveX = nondomX.copy()
    archiveF = nondomF.copy()

    if len(nondomX)>2:
        archiveX, archiveF = crowding_sorting(nondomX, nondomF)

    #Debug
    for i in range(len(archiveF)):

        var1 = costFunction(archiveX[i,:])
        var2 = archiveF[i,:]
        if debug: print var2.size


        if var1[0]!=var2[0] or var1[1]!=var2[1]:
            if debug: print '***************!!!!!!!'

    it=0
    while it < maxit:
        ### This is to see how much time is left
        if debug == False:
            pass
           # print "iters done {0}\r".format(it),
           # sys.stdout.flush()
        ###
        it = it + 1
        h = SelectLeader(archiveX,archiveF)
        w = wdamp-(wdamp-w)/maxit*it #???
        for i in xrange(0, swarmsize):

            V[i,:] = w*V[i,:] + random.random()*c1*(PXbest[i,:] - X[i,:]) + random.random()*c2*(archiveX[h,:] - X[i,:])
            X[i,:] = X[i,:] + V[i,:]

    		#:::: Check that the particles do not fly out of the search space ::::#
            X[i,X[i,:] < lb] = lb[X[i,:] < lb]#???
            X[i,X[i,:] > ub] = ub[X[i,:] > ub]
    		#:::: Change the velocity direction ::::#
            V[i,X[i,:] < lb] = -V[i,X[i,:] < lb]
            V[i,X[i,:] > ub] = -V[i,X[i,:] > ub]
    		#:::: Constrain the velocity ::::#
            V[i, V[i,:] > vmax] = vmax[V[i,:] > vmax]
            V[i, V[i,:] < -vmax] = vmax[V[i,:] < -vmax]


        # use pool.map, just as above
        if processes > 1:
            F= np.array(mp_pool.map(costFunction, X))
            if debug: print 'multi process works'
        else:
            for i in range(swarmsize):
                F[i]= costFunction(X[i, :])



        ############## Mutation
        #prepare for mutation
        pm = (1-(it-1)/maxit)**(1/mu)
        #pm = 0.3
       
        NewX = X.copy()
        NewF = F.copy()

        # create list of indices to be mutated
        mutindices = []
        for i in xrange(0, swarmsize):
            if np.random.rand() < pm :
                mutindices.append(i)
                NewX[i,:] = Mutate(X[i,:],pm,mu,lb,ub)

        # calculate cost function for mutated
        if processes > 1:
            if len(mutindices) > 0:
                NewF[mutindices,:]= np.array(mp_pool.map(costFunction, NewX[mutindices,:]))
                if debug: print 'multi process works for mutation'
        else:
            for i in mutindices:    
                NewF[i]= costFunction(NewX[i, :])                             

        ##############End of mutiation

        for i in xrange(0, swarmsize):
            if Dominates(NewF[i,:],F[i,:]):
                X[i,:] = NewX[i,:].copy()
                F[i,:] = NewF[i,:].copy()
                print 'NewF dominates F' 
            elif Dominates(F[i,:],NewF[i,:]):
                print 'F dominates NewF'                
                pass
            elif np.random.rand() < 0.5: #random.randint(0,1) == 0 can make like this also
                X[i,:] = NewX[i,:].copy() 
                F[i,:] = NewF[i,:].copy()
                print 'Nothing dominates1'
            else:
                print 'Nothing dominates2'

            if Dominates(F[i,:],PFbest[i,:]):
                PFbest[i,:] = F[i,:].copy()
                PXbest[i,:] = X[i,:].copy()
            elif Dominates(PFbest[i,:],F[i,:]):
                pass
            elif (random.randint(0,1) == 0):
                PFbest[i,:] = F[i,:].copy()
                PXbest[i,:] = X[i,:].copy()

        archiveX = np.concatenate((archiveX,X), axis = 0)
        archiveF = np.concatenate((archiveF,F), axis = 0)
        archiveX, archiveF = nondomSolutions(archiveX,archiveF) 

        if len(archiveX)>2:
            archiveX,archiveF = crowding_sorting(archiveX,archiveF)

        if len(archiveX)>nRep:
            archiveX,archiveF=removeparticles(archiveX, archiveF, len(archiveX) - nRep)
            if debug: print len(archiveX) - nRep, 'particles removed'

        if debug: print 'the',it,'th iteration finished'

        np.savetxt('archiveF.txt',archiveF)
        np.savetxt('archiveX.txt',archiveX)
    if debug == False: print
    return archiveX,archiveF
