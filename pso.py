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



class PSO:

    def __init__(self, func_name, lb, ub, bait = np.array([]), ieqcons = [], f_ieqcons = None, args = (), kwargs={},

        swarmsize = 100, omega=0.72984, phip=1.49618, phig=1.49618, maxiter = 100,

        minstep = 1e-8, minfunc = 1e-8, mu = 0.0, alpha = 2, debug = False, processes = 1,

        particle_output = False, iteration_out = False, init_dist = 'lhs', info = False, continue_flag = False):

        self.func = getattr(functions, func_name)

        self.lb = np.array(lb)

        self.ub = np.array(ub)

        self.bait = bait

        self.ieqcons = ieqcons

        self.f_ieqcons = f_ieqcons

        self.args = args

        self.kwargs = kwargs

        self.swarmsize = swarmsize

        self.omega = omega

        self.phip =  phip

        self.phig =  phig

        self.maxiter = maxiter

        self.minstep = minstep

        self.minfunc = minfunc

        self.mu = mu

        self.alpha = alpha

        self.debug = debug

        self.processes = processes

        self.particle_output = particle_output

        self.iteration_out = iteration_out

        self.init_dist = init_dist

        self.info = info

        self.x = np.random.rand(swarmsize, len(lb)) #particle positions

        self.v = np.zeros_like(self.x)  # particle velocities

        self.p = np.zeros_like(self.x)

        self.fx = np.zeros(swarmsize)  # current particle function values

        self.fs = np.zeros(swarmsize, dtype=bool)  # feasibility of each particle

        self.fp = np.ones(swarmsize)*np.inf  # best particle function values

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

        nVar = len(x)

        nMu = np.ceil(mu*nVar)

        mu_id = random.sample(range(nVar),int(nMu)) #chose particles to mutate

        sigma =  pm * (ub - lb)

        if len(sigma)>1:

            sigma = [sigma[i] for i in mu_id]

        y = x[:]

        rerror= np.random.rand(len(mu_id))*sigma

        for v in range(int(nMu)):

            y[mu_id[v]]=x[mu_id[v]]+rerror[v]

            if y[mu_id[v]] < lb[mu_id[v]]:

                y[mu_id[v]]=lb[mu_id[v]]

            if y[mu_id[v]] > ub[mu_id[v]]:

                y[mu_id[v]] = ub[mu_id[v]]

        return np.array(y)


    def sopso(self):

        """

        lb: lower boundary of variables to be optimized

        ub: upper boundary of variables to be optimized

        swarmsize: the number of particles

        maxit: maximum iterations

        nRep: the number of particles saved in the repository

        omega: final inetia factor

        phip: cognitive facotr

        phig: social learning factor

        mu: 0<= mu <=1, mutation rate. [e1,e2,e3,e4,e5,e6] is the postion one particle, ceil(mu*6) of the 6 dimensions should mutate.

        alpha: the coefficient to determin the mutation probability decaying with iterations, should be positve! Larger value makes the decaying faster.

        bait: should be a list(consisting of list),to speed up the convergency if some good settings already exisiting

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

            if self.debug: print 'load data from pickle'
            self.maxiter += self.it - 1

        assert len(self.lb)==len(self.ub), 'Lower- and upper-bounds must be the same length'

        assert hasattr(self.func, '__call__'), 'Invalid function handle'



        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'


        #Store the variables of the current run of the function in .txt or .mat file


        self.vhigh = np.abs(self.ub - self.lb)

        self.vlow = -self.vhigh



        # Initialize objective function

        obj = partial(self._obj_wrapper, self.func, self.args, self.kwargs)

        # Check for constraint function(s) #########################################

        if self.f_ieqcons is None:

            if not len(self.ieqcons):

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

        # Initialize the particle swarm ############################################

        if self.continue_flag == False:
            self.init_particle_swarm(obj, is_feasible)
            self.it = 1
        else:
            self.use_bait()
        ########### now have a choice for initial positions

        # Iterate until termination criterion met ##################################

        if self.iteration_out:
            file_iter_out = open('iteration_out.pk', 'wb')
            #self_dict = copy.deepcopy(self.__dict__)
            pickle.dump(self.__dict__, file_iter_out)


        while self.it <= self.maxiter:
            try:
                #omega = 0.7298 -(0.7298-omega)/maxiter*it

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

                    self.fx = np.array(mp_pool.map(obj, self.x))

                    self.fs = np.array(mp_pool.map(is_feasible, self.x))

                else:

                    for i in range(self.swarmsize):

                        self.fx[i] = obj(self.x[i, :])

                        self.fs[i] = is_feasible(self.x[i, :])

                # Store particle's best position (if constraints are satisfied)



               #####preparation for mutation####

                if self.mu!=0:

                    if self.debug:
                        print "doing mutations"

                    NewX = self.x.copy()

                    NewF = self.fx.copy()

                    NewFs = self.fs.copy()

                    pm = (1-(self.it-1)/self.maxiter)**(1/self.mu)

                    #pm = np.exp(-self.alpha*self.it/self.maxiter)   #pm is the mutation probability, which decays with interations#

                    mutation_idx = []

                    for i in range(0, self.swarmsize):

                        if np.random.rand() < pm:

                            mutation_idx.append(i)

                            NewX[i,:]= self.Mutate(self.x[i,:],pm,self.mu,self.lb,self.ub)



                    #print 'mutation idx',mutation_idx

                    if self.processes >1:

                        if len(mutation_idx)>0:

                            NewF[mutation_idx] = np.array(mp_pool.map(obj,NewX[mutation_idx,:]))

                            NewFs[mutation_idx] = np.array(mp_pool.map(is_feasible, NewX[mutation_idx,:]))

                    else:

                        for i in mutation_idx:

                            NewF[i] = obj(NewX[i,:])

                            NewFs[i] = is_feasible(NewX[i,:])



                    good_mutation = np.logical_and((NewF<self.fx),NewFs)

                    #print sum(good_mutation),'good mutation'

                    self.x[good_mutation,:] = NewX[good_mutation,:].copy()

                    self.fx[good_mutation] = NewF[good_mutation].copy()

                ######## mutation  finished #############



                i_update = np.logical_and((self.fx < self.fp), self.fs)

                self.p[i_update, :] = self.x[i_update, :].copy()

                self.fp[i_update] = self.fx[i_update]


                # Compare swarm's best position with global best position

                i_min = np.argmin(self.fp)

                if self.fp[i_min] < self.fg :

                    if self.debug:

                        print('New best for swarm at iteration {:}: {:} {:}'\

                            .format(self.it, self.p[i_min, :], self.fp[i_min]))



                    p_min = self.p[i_min, :].copy()

                    stepsize = np.sqrt(np.sum((self.g - p_min)**2))



                    if np.abs(self.fg - self.fp[i_min]) <= self.minfunc:

                        self.it += 1

                        if (self.info == True): print('Stopping search: Swarm best objective change less than {:}'.format(self.minfunc))

                        if self.particle_output:

                            return p_min, self.fp[i_min], self.p, self.fp

                        else:

                            return p_min, self.fp[i_min]

                    elif stepsize <= self.minstep:

                        self.it += 1

                        if (self.info == True): print('Stopping search: Swarm best position change less than {:}'.format(self.minstep))

                        if self.particle_output:

                            return p_min, self.fp[i_min], self.p, self.fp

                        else:

                            return p_min, self.fp[i_min]

                    else:

                        self.g = p_min.copy()

                        self.fg = self.fp[i_min]



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
                raise

            if (self.iteration_out):
                pickle.dump(self.__dict__, file_iter_out)
                file_iter_out.flush()                           #When it closes?

            #Pickle all info about current iteration. This way is quite slow, but we can ignore it in comparision with time for function computing
            with open('info_of_last_run.pk', 'wb') as file_info_of_last:
                pickle.dump(self.__dict__, file_info_of_last)
        ###end of BIG while

        if self.iteration_out:
            file_iter_out.close()

        if (self.info == True):
            print('Stopping search: maximum iterations reached --> {:}'.format(self.maxiter))

        if not is_feasible(self.g):

            if (self.info == True): print("However, the optimization couldn't find a feasible design. Sorry")

        if self.particle_output:

            return self.g, self.fg, self.p, self.fp

        else:

	        return self.g, self.fg

    def init_particle_swarm(self, obj, is_feasible):

        if (self.init_dist == 'rand'):

	        self.x = np.random.rand(self.swarmsize, self.dim)  # particle positions

        elif(self.init_dist == 'lhs'):

            self.x = self.lhs()

        # Initialize the particle's position

        self.x = self.lb + self.x*(self.ub - self.lb)

        self.use_bait()

        # Calculate objective and constraints for each particle

        if self.processes > 1:

            self.fx = np.array(mp_pool.map(obj, self.x))

            self.fs = np.array(mp_pool.map(is_feasible, self.x))

        else:

            for i in range(self.swarmsize):

                self.fx[i]= obj(self.x[i, :])

                self.fs[i] = is_feasible(self.x[i, :])

        # Store particle's best position (if constraints are satisfied)

        i_update = np.logical_and(self.fx < self.fp, self.fs)

        self.p[i_update, :] = self.x[i_update, :].copy()

        self.fp[i_update] = self.fx[i_update]

        # Update swarm's best position

        i_min = np.argmin(self.fp)

        if self.fp[i_min] < self.fg :

            self.fg = self.fp[i_min]

            self.g = self.p[i_min, :].copy()

        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            self.g = self.x[0, :].copy()

        # Initialize the particle's velocity

        self.v = self.vlow + np.random.rand(self.swarmsize, self.dim)*(self.vhigh - self.vlow)

        #self.v = self.vlow + lhs(S,D)*(self.vhigh - self.vlow)
    def use_bait(self):
        if (self.bait != np.array([])):
            assert self.dim == self.bait.shape[1]

            self.swarmsize += len(self.bait)

            self.x = np.append(self.x, self.bait, axis = 0)

            self.v = np.append(self.v, np.random.rand(len(self.bait), self.dim), axis = 0)  # particle velocities

            self.p = np.append(self.p, np.zeros_like(self.bait), axis = 0)

            self.fx = np.append(self.fx, np.zeros(len(self.bait)))  # current particle function values

            self.fs = np.append(self.fs, np.zeros(len(self.bait), dtype=bool))  # feasibility of each particle

            self.fp = np.append(self.fp, np.ones(len(self.bait))*np.inf)

            if self.debug: print 'bait has been used!'



    #For .mat handling

    def get_dict_from_class(self):

        d = copy.deepcopy(self.__dict__)

        #this is for .mat handling

        d["func"] = self.func.__name__

        for key in d:

            if d[key] is None:

                d[key] = "None"

        return d

    def save_info_of_last_run_to_mat(self):

        import scipy.io as sio

        sio.savemat('info_of_last_run.mat', self.get_dict_from_class(), oned_as='row')

    def load_mat_to_dict(self):

        import scipy.io as sio

        d_load = sio.loadmat("info_of_last_run.mat")

        new_d = {}

        for key, value in d_load.items():

            if not key.startswith('__') and not callable(key):

                value = value.tolist()

                if (len(value) == 0):

                    new_d[key] = value

                elif (len(value[0]) == 1 and (type(value[0][0]) == int or type(value[0][0]) == float)):

                    new_d[key] = value[0][0]

                elif (len(value[0]) > 1 and (type(value[0][0]) == int or type(value[0][0]) == float)):

                    new_d[key] = value[0]

                elif (isinstance(value[0], unicode)):

                    if (value[0].encode('ascii') == 'None'):

                        new_d[key] = None

                    else:

                        new_d[key] = value[0].encode('ascii')

                elif (key == 'kwargs'):

                    new_d[key] = {}

                else:

                    print ".mat data is incorreect"

                if (key == 'args'):

                    new_d[key] = ()

                if (key == 'func' and isinstance(value[0], unicode)):

                    new_d[key] = getattr(functions, value[0])

        return new_d

    def load_info_of_last_run_from_mat(self):

        self.__dict__ = self.load_mat_to_dict()

    def get_dict_of_particles_pos(self, fx, fp, x, p):

        d = OrderedDict()

        d['f(x)'] = fx

        d['f(PBx)'] = fp

        for i in range(self.dim):

            d['x_' + str(i)] = x[:, i] #here i is referred to ith coordinate and x[:, i] is x_i coordinates of S particles

        for i in range(self.dim):

            d['PBx_' + str(i)] = p[:, i]

        return d
