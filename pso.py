"""
Created on August 2018

@author: I.Fatkhullin, J.Li HZB
"""

from __future__ import division
from functools import partial
import numpy as np
import time, sys, os, time
import scipy.stats, random
import scipy.io as sio
import pandas as pd
from collections import OrderedDict
import copy
from myfunc import functions
import pickle
#for animation
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#
class PSO:

    def __init__(self, func_name, lb, ub, Nobj = 1, nRep = 100, bait = np.array([]), w = [0.72984, 0.4], ieqcons = [], f_ieqcons = None, args = (), kwargs={},
        swarmsize = 100, phip=1.49618, phig=1.49618, maxiter = 100,
        minstep = 1e-8, minfunc = 1e-8, mu = 0.0, debug = False, processes = 1,
        particle_output = False, iteration_out = False, init_dist = 'lhs', omega_strategy = 'constant', info = False, continue_flag = False):
        self.func = getattr(functions, func_name)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.Nobj = Nobj
        self.nRep = nRep
        self.bait = bait
        self.ieqcons = ieqcons
        self.f_ieqcons = f_ieqcons
        self.args = args
        self.kwargs = kwargs
        self.swarmsize = swarmsize
        self.omega_strategy = omega_strategy
        self.omega = w[0] #w - current inertia factor
        #Recommended value for SOPSO is omega=0.72984
        self.w = w #initial and final inertia factors
        self.phip =  phip #c1
        self.phig =  phig #c2
        self.maxiter = maxiter #maxit
        self.minstep = minstep #now is used only for sopso
        self.minfunc = minfunc #now is used only for sopso
        self.mu = mu
        self.debug = debug
        self.processes = processes
        self.particle_output = particle_output
        self.iteration_out = iteration_out
        self.init_dist = init_dist
        self.info = info
        self.x = np.random.rand(swarmsize, len(lb)) #particle positions
        self.v = np.zeros_like(self.x)  # particle velocities
        self.p = np.zeros_like(self.x)
        self.fx = np.zeros((swarmsize, Nobj)) # current particle function values
        self.fs = np.zeros(swarmsize, dtype=bool)  # feasibility of each particle
        self.fp = np.ones((swarmsize, Nobj))*np.inf  # best particle function values
        self.g = []  # best swarm position
        self.fg = np.inf  # best swarm position starting value
        self.it = 0
        self.continue_flag = continue_flag
        self.dim = len(lb)
    def lhs(self):
        samples = self.swarmsize
        rng = np.random
        segsize = 1.0 / samples
        rdrange = rng.rand(samples, self.dim) * segsize
        rdrange += np.atleast_2d(np.linspace(0.0, 1.0,samples, endpoint = False)).T
        x = np.zeros_like(rdrange)
        for j in range(self.dim):
            order = rng.permutation(range(samples))
            x[:,j] = rdrange[order,j]
        return x

    def _obj_wrapper(self, func, args, kwargs, x):
        return func(x, *args, **kwargs)

    def _is_feasible_wrapper(self, func, x):
        return np.all(self.func(x)>=0)

    def _cons_none_wrapper(self, x):
        return np.array([0])

    def _cons_ieqcons_wrapper(self, x):
        return np.array([y(x, *self.args, **self.kwargs) for y in self.ieqcons])

    def _cons_f_ieqcons_wrapper(self, x):
        return np.array(self.f_ieqcons(x, *self.args, **self.kwargs))

    def Mutate(self, x, pm, mu, lb, ub):
        x = list(x)
        mu_idx = np.random.randint(len(x))
        dx =  pm * (ub[mu_idx] - lb[mu_idx])
        x1 = x[mu_idx] - dx
        x2 = x[mu_idx] + dx
        if x1 < lb[mu_idx]:
            x1 = lb[mu_idx]
        if x2 > ub[mu_idx]:
            x2 = ub[mu_idx]
        x[mu_idx] = x1 + (x2 - x1)*np.random.random()
        return np.array(x)

    def Dominates(self, X, Y):
        b = False
        if all(X<=Y) and any(X<Y):
            b = True
        return b

    def nondomSolutions(self, X, F):
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

    def inverse_permutation_crowding(self, p):
        #This function computes an "inverse permutation"
        return np.array([p.index(l) for l in range(len(p))])

    def crowding_sorting(self, archiveX, archiveF):
        nondomN = len(archiveF)
        crowdDist = np.zeros((nondomN, ))
        for i in range(self.Nobj):
            indexes = np.argsort(archiveF[:,i])
            archivetestF = archiveF[indexes, i]
            crowdDist = crowdDist[indexes] #(*)
            for j in range(1, nondomN-1):
                crowdDist[j] = crowdDist[j] + (archivetestF[j+1]-archivetestF[j-1])/(archivetestF[-1]-archivetestF[0])
            crowdDist[0] = np.inf
            crowdDist[-1] = np.inf
            #at the end of each loop apply an inverse permutation to crowdDist,
            #because here (*) we should apply a permutation to initailly ordered array
            invert_indexes = self.inverse_permutation_crowding(list(indexes))
            crowdDist = crowdDist[invert_indexes]

        indx = np.argsort(crowdDist)[::-1]
        archiveX = archiveX[indx]
        archiveF = archiveF[indx]
        return archiveX, archiveF

    def removeparticles(self, archiveX,archiveF,N):
    	for i in range(int(N)):
    		archiveX=np.delete(archiveX,-1,axis=0)
    		archiveF=np.delete(archiveF,-1,axis=0)
    	return archiveX,archiveF

    def SelectLeader(self, archiveX, archiveF):
    	Nx = np.floor(len(archiveX)*0.1)
    	h = random.randint(0, Nx)
    	return h

    def weight_strategy(self):
        if(self.omega_strategy == 'chaotic'):
            #Chaotic_inertia_weight
            z = random.random()
            z = 4*z*(1-z)
            return (self.w[0]-self.w[1])*float(self.maxiter - self.it)/float(self.maxiter) + self.w[1]*z
        elif(self.omega_strategy == 'random'):
            #Random_inertia_weight
            return 0.5 + random.random()/2.0
        elif(self.omega_strategy == 'linear'):
            #Linear_inertia_weight
            return float(self.w[0]) - (self.w[0]-self.w[1])/float(self.maxiter)*float(self.it)
        elif(self.omega_strategy == 'constant'):
            #Constant_inertia_weight
            return self.w[0]
        else:
            print 'Specify omega_strategy correctly!'
            return self.w[0]

    def mopsocd(self):

        assert len(self.lb)==len(self.ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(self.func, '__call__'), 'Invalid function handle'
        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'

        ##################initialization
        costFunction = partial(self._obj_wrapper, self.func, self.args, self.kwargs)
        vmax = (self.ub-self.lb)

        # Initialize the multiprocessing module if necessary
        if self.processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(self.processes)

        self.init_particle_swarm(costFunction, ())
        if self.debug: print 'the 0th iteration finished'

        #########################initialization is finished
        nondomX, nondomF = self.nondomSolutions(self.x, self.fx)
        archiveX = nondomX.copy()
        archiveF = nondomF.copy()
        if len(nondomX)>2:
            archiveX, archiveF = self.crowding_sorting(nondomX, nondomF)

        self.it = 0
        while self.it < self.maxiter:

            ### This is to see how much time is left (first part)
            #print "iters done {0}\r".format(it),
            #sys.stdout.flush()
            ###
            self.it = self.it + 1
            h = self.SelectLeader(archiveX, archiveF)

            self.omega = self.weight_strategy() #updates omega if it is not constant

            for i in range(self.swarmsize):
                # Update the particles velocities
                self.v[i,:] = self.omega*self.v[i,:] + random.random()*self.phip*(self.p[i,:] - self.x[i,:]) + random.random()*self.phig*(archiveX[h,:] - self.x[i,:])
                # Update the particles' positions
                self.x[i,:] = self.x[i,:] + self.v[i,:]
        		#:::: Check that the particles do not fly out of the search space ::::#
                self.x[i,self.x[i,:] < self.lb] = self.lb[self.x[i,:] < self.lb]
                self.x[i,self.x[i,:] > self.ub] = self.ub[self.x[i,:] > self.ub]
        		#:::: Change the velocity direction ::::#
                self.v[i,self.x[i,:] < self.lb] = -self.v[i,self.x[i,:] < self.lb]
                self.v[i,self.x[i,:] > self.ub] = -self.v[i,self.x[i,:] > self.ub]
        		#:::: Constrain the velocity ::::#
                self.v[i, self.v[i,:] > vmax] = vmax[self.v[i,:] > vmax]
                self.v[i, self.v[i,:] < -vmax] = vmax[self.v[i,:] < -vmax]

            if self.processes > 1:
                self.fx = np.array(mp_pool.map(costFunction, self.x))
                if self.debug: print 'multi process works'
            else:
                for i in range(self.swarmsize):
                    self.fx[i] = costFunction(self.x[i, :])

            ############## Mutation
            #prepare for mutation
            if (self.mu == 0.0):
                self.mu = 0.05
                print 'In multiobjective optimization one needs mutation. Set mu = 0.05'

            pm = (1-(float(self.it))/float(self.maxiter))**(1/self.mu)


            NewX = self.x.copy()
            NewF = self.fx.copy()

            # create list of indices to be mutated
            mutindices = []
            for i in xrange(0, self.swarmsize):
                if np.random.rand() < pm :
                    mutindices.append(i)
                    NewX[i,:] = self.Mutate(self.x[i,:], pm, self.mu, self.lb, self.ub)

            # calculate cost function for mutated
            if self.processes > 1:
                if len(mutindices) > 0:
                    NewF[mutindices,:]= np.array(mp_pool.map(costFunction, NewX[mutindices,:]))
                    if self.debug: print 'multi process works for mutation'
            else:
                for i in mutindices:
                    NewF[i] = costFunction(NewX[i, :])
            ##############End of mutiation

            for i in xrange(0, self.swarmsize):
                if self.Dominates(NewF[i,:], self.fx[i,:]):
                    self.x[i,:] = NewX[i,:].copy()
                    self.fx[i,:] = NewF[i,:].copy()
                elif self.Dominates(self.fx[i,:], NewF[i,:]):
                    pass
                elif random.randint(0,1) == 0:
                    self.x[i,:] = NewX[i,:].copy()
                    self.fx[i,:] = NewF[i,:].copy()
                #Selecting (changing) personal best
                if self.Dominates(self.fx[i,:], self.fp[i,:]):
                    self.fp[i,:] = self.fx[i,:].copy()
                    self.p[i,:] = self.x[i,:].copy()
                elif self.Dominates(self.fp[i,:], self.fx[i,:]):
                    pass
                elif (random.randint(0,1) == 0):
                    self.fp[i,:] = self.fx[i,:].copy()
                    self.p[i,:] = self.x[i,:].copy()

            archiveX = np.concatenate((archiveX,self.x), axis = 0)
            archiveF = np.concatenate((archiveF,self.fx), axis = 0)
            archiveX, archiveF = self.nondomSolutions(archiveX,archiveF)

            if len(archiveX)>2:
                archiveX, archiveF = self.crowding_sorting(archiveX,archiveF)

            if len(archiveX) > self.nRep:
                archiveX, archiveF = self.removeparticles(archiveX, archiveF, len(archiveX) - self.nRep)
                if self.debug: print len(archiveX) - self.nRep, 'particles removed'

            if self.debug: print 'the', self.it, 'th iteration finished'

            np.savetxt('archiveF.txt', archiveF)
            np.savetxt('archiveX.txt', archiveX)


        ###This is to see how much time is left (second part)
        #print
        return archiveX, archiveF


    def sopso(self):
        """
        func or costFunction: optimized function, might be SO or MO
        lb: (list) lower boundary of variables to be optimized
        ub: (list) upper boundary of variables to be optimized
        swarmsize: the number of particles
        maxiter: maximum iterations
        nRep: the number of particles saved in the repository
        omega: current inetia factor
        w = [0.9, 0.4]: initial and final inertia factors
        phip: cognitive facotor
        phig: social learning factor
        mu: 0 <= mu <= 1, mutation rate. [e1,e2,e3,e4,e5,e6] is the postion one particle, ceil(mu*6) of the 6 dimensions should mutate.
        bait: to speed up the convergency if some good settings already exisiting, adds particles from bait to exisiting swarm
        processes: for multiprocesses
        """

        if self.continue_flag:
            with open('info_of_last_run.pk', 'rb') as file_info_of_last:
                old_dic = pickle.load(file_info_of_last)
            self.__dict__['x'] = old_dic['x']
            self.__dict__['v'] = old_dic['v']
            self.__dict__['p'] = old_dic['p']
            self.__dict__['fx'] = old_dic['fx']
            self.__dict__['fs'] = old_dic['fs']
            self.__dict__['fp'] = old_dic['fp']
            self.__dict__['g'] = old_dic['g']
            self.__dict__['fg'] = old_dic['fg']
            self.__dict__['swarmsize'] = old_dic['swarmsize']
            self.__dict__['it'] = old_dic['it']
            self.__dict__['omega'] = old_dic['omega']
            if self.debug: print 'load data from pickle'
            self.maxiter += self.it - 1

        assert len(self.lb)==len(self.ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(self.func, '__call__'), 'Invalid function handle'
        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'

        # Initialize objective function
        costFunction = partial(self._obj_wrapper, self.func, self.args, self.kwargs)

        # Check for constraint function(s) #########################################
        if self.f_ieqcons is None:
            if len(self.ieqcons) == 0:
                if self.debug:
                    print('No constraints given.')
                cons = self._cons_none_wrapper
            else:
                if self.debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(self._cons_ieqcons_wrapper, self.ieqcons, self.args, self.kwargs)
        else:
            if self.debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(self._cons_f_ieqcons_wrapper, self.f_ieqcons, self.args, self.kwargs)
        is_feasible = partial(self._is_feasible_wrapper, cons)

        # Initialize the multiprocessing module if necessary
        if self.processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(self.processes)

        ############ Initialize the particle swarm ############################################
        if self.continue_flag == False:
            self.init_particle_swarm(costFunction, is_feasible)
            self.it = 1
        else:
            self.use_bait()
        ########### Initialization is finished#############

        if self.iteration_out:
            file_iter_out = open('iteration_out.pk', 'wb')
            pickle.dump(self.__dict__, file_iter_out)

        ############# Iterate until termination criterion met ##################################
        while self.it <= self.maxiter:
            ### This is to see how much time is left
             #  print "iters done {0}\r".format(it),
            #   sys.stdout.flush()
            ###

            try:
                #Selection of inertia weight strategy
                self.omega = self.weight_strategy()

                rp = np.random.uniform(size=(self.swarmsize, self.dim))
                rg = np.random.uniform(size=(self.swarmsize, self.dim))
                # Update the particles velocities
                self.v = self.omega*self.v + self.phip*rp*(self.p - self.x) + self.phig*rg*(self.g - self.x)
                # Update the particles' positions
                self.x = self.x + self.v

                # Correct for bound violations
                maskl = self.x < self.lb
                masku = self.x > self.ub
                self.x = self.x*(~np.logical_or(maskl, masku)) + self.lb*maskl + self.ub*masku

                # Update objectives and constraints
                if self.processes > 1:
                    self.fx[:,0] = np.array(mp_pool.map(costFunction, self.x))
                    self.fs = np.array(mp_pool.map(is_feasible, self.x))
                else:
                    for i in range(self.swarmsize):
                        self.fx[i, 0] = costFunction(self.x[i, :])
                        self.fs[i] = is_feasible(self.x[i, :])

                #Update personal best
                i_update = np.logical_and((self.fx[:,0] < self.fp[:,0]), self.fs)
                self.p[i_update, :] = self.x[i_update, :].copy()
                self.fp[i_update, 0] = self.fx[i_update,0]

               #####preparation for mutation####
                if self.mu != 0.0:
                    NewX = self.x.copy()
                    NewF = self.fx[:,0].copy()
                    NewFs = self.fs.copy()
                    pm = (1-(self.it-1)/float(self.maxiter))**(1/self.mu) #pm is the mutation probability, which decays with interations#
                    mutation_idx = []
                    for i in range(0, self.swarmsize):
                        if np.random.rand() < pm:
                            mutation_idx.append(i)
                            NewX[i,:]= self.Mutate(self.x[i,:],pm,self.mu,self.lb,self.ub)

                    if self.processes > 1:
                        if len(mutation_idx)>0:
                            NewF[mutation_idx] = np.array(mp_pool.map(costFunction,NewX[mutation_idx,:]))
                            NewFs[mutation_idx] = np.array(mp_pool.map(is_feasible, NewX[mutation_idx,:]))
                    else:
                        for i in mutation_idx:
                            NewF[i] = costFunction(NewX[i,:])
                            NewFs[i] = is_feasible(NewX[i,:])

                    #new strategy
                    self.x = copy.deepcopy(NewX)
                    self.fx[:,0] = copy.deepcopy(NewF)
                ######## mutation is finished #############

                #Update personal best
                i_update = np.logical_and((self.fx[:,0] < self.fp[:,0]), self.fs)
                self.p[i_update, :] = self.x[i_update, :].copy()
                self.fp[i_update, 0] = self.fx[i_update,0]

                #Compare swarm's best position with global best position
                i_min = np.argmin(self.fp[:,0])
                if self.fp[i_min,0] < self.fg :
                    if self.debug:
                        print('New best for swarm at iteration {:}: {:} {:}'\
                            .format(self.it, self.p[i_min, :], self.fp[i_min,0]))
                    p_min = self.p[i_min, :].copy()
                    stepsize = np.sqrt(np.sum((self.g - p_min)**2))
                    if np.abs(self.fg - self.fp[i_min,0]) <= self.minfunc:
                        self.it += 1
                        if (self.info == True): print('Stopping search: Swarm best objective change less than {:}'.format(self.minfunc))
                        if self.particle_output:
                            return p_min, self.fp[i_min,0], self.p, self.fp[:,0]
                        else:
                            return p_min, self.fp[i_min,0]
                    elif stepsize <= self.minstep:
                        self.it += 1
                        if (self.info == True): print('Stopping search: Swarm best position change less than {:}'.format(self.minstep))
                        if self.particle_output:
                            return p_min, self.fp[i_min,0], self.p, self.fp[:,0]
                        else:
                            return p_min, self.fp[i_min,0]
                    else:
                        self.g = p_min.copy()
                        self.fg = self.fp[i_min,0]

                if self.debug:
                    print 'Best after iteration {:}: {:} {:}'.format(self.it, self.g, self.fg)

                self.it += 1

            except KeyboardInterrupt or SystemError:
                try:
                    with open('info_of_last_run.pk', 'rb') as file_info_of_last:
                        self.__dict__ = pickle.load(file_info_of_last)
                    with open('info_of_last_run.pk', 'wb') as file_info_of_last:
                        pickle.dump(self.__dict__, file_info_of_last)
                except EOFError:
                    pass
                if self.iteration_out:
                    file_iter_out.close()
                raise #close the file and then raise  EOFerror

            if (self.iteration_out):
                pickle.dump(self.__dict__, file_iter_out)
                file_iter_out.flush()

            #Pickle all info about current iteration. This way is quite slow, but we can ignore it in comparision with time for function computing
            with open('info_of_last_run.pk', 'wb') as file_info_of_last:
                pickle.dump(self.__dict__, file_info_of_last)


        ###############################end of BIG while
        ### This is to see how much time is left (second part)
        #print
        ###


        if self.iteration_out:
            file_iter_out.close()
        if (self.info == True):
            print('Stopping search: maximum iterations reached --> {:}'.format(self.maxiter))
        if not is_feasible(self.g):
            if (self.info == True): print("However, the optimization couldn't find a feasible design. Sorry")
        if self.particle_output:
            return self.g, self.fg, self.p, self.fp[:,0]
        else:
	        return self.g, self.fg

    def init_particle_swarm(self, costFunction, is_feasible):
        if (self.init_dist == 'rand'):
	        self.x = np.random.rand(self.swarmsize, self.dim)  # particle positions
        elif(self.init_dist == 'lhs'):
            self.x = self.lhs()
        # Initialize the particle's position
        self.x = self.lb + self.x*(self.ub - self.lb)
        self.use_bait()
        # Calculate objective and constraints for each particle
        if self.processes > 1:
            if self.Nobj == 1:
                self.fs = np.array(mp_pool.map(is_feasible, self.x))
                self.fx[:,0] = np.array(mp_pool.map(costFunction, self.x))
            else:
                self.fx = np.array(mp_pool.map(costFunction, self.x))
            if self.debug: print 'multiprocessing works'
        else:
            for i in range(self.swarmsize):
                if self.Nobj == 1:
                    self.fx[i,0] = costFunction(self.x[i, :])
                    self.fs[i] = is_feasible(self.x[i, :])
                else:
                    self.fx[i] = costFunction(self.x[i, :])

        if self.Nobj == 1:
            # Store particle's best position (if constraints are satisfied)

            i_update = np.logical_and(self.fx[:,0] < self.fp[:,0], self.fs)

            self.p[i_update, :] = self.x[i_update, :].copy()
            self.fp[i_update,0] = self.fx[i_update,0]
            # Update swarm's best position
            i_min = np.argmin(self.fp[:,0])
            if self.fp[i_min,0] < self.fg :
                self.fg = self.fp[i_min,0]
                self.g = self.p[i_min, :].copy()
            else:
                # At the start, there may not be any feasible starting point, so just
                # give it a temporary "best" point since it's likely to change
                self.g = self.x[0, :].copy()
        else:
            #Initialize personal best by initial positions
            self.p = self.x.copy()
            self.fp[:,0] = self.fx[:,0].copy()

        # Initialize the particle's velocity
        vhigh = np.abs(self.ub - self.lb)
        vlow = -vhigh
        self.v = vlow + np.random.rand(self.swarmsize, self.dim)*(vhigh - vlow)

    def use_bait(self):
        if (self.bait != np.array([])):
            assert self.dim == self.bait.shape[1]
            self.swarmsize += len(self.bait)
            self.x = np.append(self.x, self.bait, axis = 0)
            self.v = np.append(self.v, np.random.rand(len(self.bait), self.dim), axis = 0)  # particle velocities
            self.p = np.append(self.p, self.bait, axis = 0)
            self.fx = np.append(self.fx, np.zeros((len(self.bait), self.Nobj)), axis = 0)  # current particle function values
            if self.Nobj == 1:
                self.fs = np.append(self.fs, np.zeros(len(self.bait), dtype=bool))  # feasibility of each particle
            self.fp = np.append(self.fp, np.ones((len(self.bait), self.Nobj))*np.inf, axis = 0)

            if self.debug: print 'bait has been used!'
