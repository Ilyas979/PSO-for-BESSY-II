from __future__ import division
from functools import partial
import numpy as np
import time,sys,os,time
import scipy.stats,random
import scipy.io as sio
import pandas as pd
from collections import OrderedDict
#from pyDOE import *
import copy
from myfunc import functions
import pickle

class PSO:
    def __init__(self, func_name, lb, ub, bait = [], vel = [], ieqcons = [], f_ieqcons = None, args = (), kwargs={},
        swarmsize = 100, omega = 0.4, phip = 0.5, phig = 0.5, maxiter = 100,
        minstep = 1e-8, minfunc = 1e-8, mu = 0.0, alpha = 2, debug = False, processes = 1,
        particle_output = False, iteration_out = False, init_pos = 'lhs', info = False):
        self.func = getattr(functions, func_name)
        self.lb = lb
        self.ub = ub        
        self.bait = bait        
        self.vel = vel
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
        self.init_pos = init_pos
        self.info = info

    @classmethod
    def extractall(cls):        
        print  {key:value for key, value in cls.__dict__.items() if not key.startswith('__') and not callable(key)}

    def lhs(self):
        samples = self.swarmsize
        N = len(self.lb)
        rng = np.random

        segsize = 1.0 / samples

        rdrange = rng.rand(samples, N) * segsize
        rdrange += np.atleast_2d(np.linspace(0.0, 1.0,samples, endpoint = False)).T

        x = np.zeros_like(rdrange)

        for j in range(N):
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
        mu_id = random.sample(range(nVar),int(nMu))

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
   
    def continue_unfinished_pso(self, iteration_out_csv = 'iteration_out.csv', it_info = 'it_info.txt', more_iters = 100, continue_previous_iters = False, get_var_from_mat = True):
        if get_var_from_mat: self.mat_to_class()
        it = self.maxiter
        if (continue_previous_iters):        #"it" is an iter in which it stopped the last time
            run_info = open(it_info, "r")
            it_strs = run_info.read().split(' ')
            it = int(it_strs[0])
            run_info.close()
            if len(it_strs) == 1:
                if self.info: print "The last run was successul. We are going to continue now"        
            elif len(it_strs) == 2:           #These never happens now
                if int(it_str[1]) == -1:
                    print ('PSO has finished the last time: Swarm best objective change was less than {:}'.format(self.minfunc) + ' it is better to start again with pso')
                    return np.inf, np.inf
                    
                elif int(it_strs[1]) == -2:
                    print ('PSO has finished the last time: Swarm best position change was less than {:}'.format(self.minstep) + ' it is better to start again with pso')
                    return np.inf, np.inf
                elif int(it_strs[1]) < -2 or int(it_str[1]) == 0:
                    print 'Something was wrong in recording data the last time'
                    return np.inf, np.inf
                else:
                    print 'Something was wrong in recording data the last time'
                    return np.inf, np.inf
            else:
                print 'the last run was really bad'
                return np.inf, np.inf
        
        df = pd.read_csv(iteration_out_csv)
        D = len(self.lb)
        x_cols = []
        for i in range(D):
            x_cols.append('x_' + str(i))

        #initial velocities
        ###########
        x1 = df[x_cols].iloc[:self.swarmsize].values.tolist()
        x2 = df[x_cols].iloc[self.swarmsize:].values.tolist()
        for i in range(len(x2)):
            for j in range(len(x2[0])):
	            x2[i][j] -= x1[i][j]

        ###########


        self.bait = df[x_cols].iloc[self.swarmsize:].values.tolist()
        self.vel = x2
        self.maxiter = self.maxiter - it + more_iters
        xopt, fopt = self.pso()
        return xopt, fopt
    
    def save_info_of_last_run_to_txt(self): #one of the options for developers
        with open("info_of_last_run.txt", "w+") as run_info:
            run_info.write(self.func.__name__ + '\n')
            run_info.write(','.join([str(i) for i in self.lb]) + '\n')
            run_info.write(','.join([str(i) for i in self.ub]) + '\n')
            run_info.write(','.join([str(i) for i in self.ieqcons]) + '\n')
            run_info.write(str(self.f_ieqcons) + '\n')
            run_info.write(str(self.swarmsize) + '\n')
            run_info.write(str(self.omega) + '\n')
            run_info.write(str(self.phip) + '\n')
            run_info.write(str(self.phig) + '\n')
            run_info.write(str(self.maxiter) + '\n')
            run_info.write(str(self.minstep) + '\n')
            run_info.write(str(self.minfunc) + '\n')
            run_info.write(str(self.mu) + '\n')
            run_info.write(str(self.alpha) + '\n')
            run_info.write(str(self.debug) + '\n')
            run_info.write(str(self.processes) + '\n')
            run_info.write(str(self.particle_output) + '\n')
            run_info.write(str(self.iteration_out) + '\n')
            run_info.write(str(self.init_pos) + '\n')
            run_info.write(str(self.info) + '\n')
            #size_of_run_info = run_info.tell()

    def get_dict_from_class(self):
        d = copy.deepcopy(self.__dict__)
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

    def mat_to_class(self):
        self.__dict__ = self.load_mat_to_dict()        

    def pso(self):
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
        
      
        assert len(self.lb)==len(self.ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(self.func, '__call__'), 'Invalid function handle'
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'


        #Store the variables of the current run of the function in .txt or .mat file

        
        #self.save_info_of_last_run_to_txt()
        
        self.save_info_of_last_run_to_mat()
        

        it_info = open("it_info.txt", "w+", buffering = 0)

        vhigh = np.abs(self.ub - self.lb)
        vlow = -vhigh

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
        S = self.swarmsize
        D = len(self.lb)  # the number of dimensions each particle has

        ########### now have a choice for initial positions
        if (self.init_pos == 'rand'):
	        x = np.random.rand(S, D)  # particle positions
        elif(self.init_pos == 'lhs'):
            x = self.lhs()
            #x = lhs(D, samples = S)

        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values

        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value

        # Initialize the particle's position
        x = self.lb + x*(self.ub - self.lb)
	    #x = creat_iniswarm(lb,ub,S,D)
        if (self.bait != []):
            idxx = range(len(self.bait))
            for idx, elem in zip(idxx, self.bait):
                x[idx,:] = np.array(elem)
	        if self.debug: print 'bait has been used!'

        # Calculate objective and constraints for each particle
        if self.processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
               # print 'the ',i,'th element in 0th iteration'
                fx[i]= obj(x[i, :])
                fs[i] = is_feasible(x[i, :])


        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and(fx < fp, fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]


        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg :
            fg = fp[i_min]
            g = p[i_min, :].copy()

        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()



        # regardless of the value of "iteration_out" will save at least 2 last iters ##################iteration_out######
        d = OrderedDict()
        d['f(x)'] = fx
        d['f(PBx)'] = fp
        for i in range(D):
            d['x_' + str(i)] = x[:, i] #here i is referred to ith coordinate and x[:, i] is x_i coordinates of S particles
        for i in range(D):
            d['PBx_' + str(i)] = p[:, i]

        d_old = copy.deepcopy(d)
        df = pd.DataFrame(data = d, columns = d.keys())
        df.to_csv('iteration_out.csv', header = True)



        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
        v = 0.4*v
        #v = vlow + lhs(S,D)*(vhigh - vlow)
        if (self.vel != []):
            idxx = range(len(self.vel))
            for idx, elem in zip(idxx, self.vel):
                v[idx,:] = np.array(elem)
            if self.debug: print 'vel has been used!'

        # Iterate until termination criterion met ##################################
        it = 1

        while it <= self.maxiter:
            #omega = 0.7298 -(0.7298-omega)/maxiter*it
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))

            # Update the particles velocities
            v = self.omega*v + self.phip*rp*(p - x) + self.phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < self.lb
            masku = x > self.ub
            x = x*(~np.logical_or(maskl, masku)) + self.lb*maskl + self.ub*masku


            # Update objectives and constraints
            if self.processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
            # Store particle's best position (if constraints are satisfied)




           #####preparation for mutation####
            if self.mu!=0:
                print "doing mutations"
                NewX = x.copy()
                NewF = fx.copy()
                NewFs = fs.copy()
                pm = (1-(it-1)/self.maxiter)**(1/self.mu)
                pm = np.exp(-self.alpha*it/self.maxiter)   #pm is the mutation probability, which decays with interations#
                mutation_idx = []
                for i in range(0, S):
                    if np.random.rand()<pm:
                        mutation_idx.append(i)
                        NewX[i,:]= self.Mutate(x[i,:],pm,self.mu,self.lb,self.ub)

                #print 'mutation idx',mutation_idx
                if self.processes >1:
                    if len(mutation_idx)>0:
                        NewF[mutation_idx] = np.array(mp_pool.map(obj,NewX[mutation_idx,:]))
                        NewFs[mutation_idx] = np.array(mp_pool.map(is_feasible, NewX[mutation_idx,:]))
                else:
                    for i in mutation_idx:
                        NewF[i] = obj(NewX[i,:])
                        NewFs[i] = is_feasible(NewX[i,:])

                good_mutation = np.logical_and((NewF<fx),NewFs)
                #print sum(good_mutation),'good mutation'
                x[good_mutation,:] = NewX[good_mutation,:].copy()
                fx[good_mutation] = NewF[good_mutation].copy()
            ######## mutation  finished #############




            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]

            d = OrderedDict()
            d['f(x)'] = fx
            d['f(PBx)'] = fp
            for i in range(D):
                d['x_' + str(i)] = x[:, i]
            for i in range(D):
                d['PBx_' + str(i)] = p[:, i]

            '''
            if (self.iteration_out):
                df = pd.DataFrame(data = d, columns = d.keys())
                df.to_csv('iteration_out.csv', mode = 'a', header = False)
            else:
                df_old = pd.DataFrame(data = d_old, columns = d_old.keys())
                df_old.to_csv('iteration_out.csv', mode = 'w', header = True)
                df = pd.DataFrame(data = d, columns = d.keys())
                df.to_csv('iteration_out.csv', mode = 'a', header = False)
                d_old = copy.deepcopy(d)
            '''
            #Pickle
            if (self.iteration_out):
                f = open('iteration_out.pk', 'wb') #check what is wb
                pickle.dump(d, f)
                f.close()
            else:
                while 1:
                    try:
                        pickle.load(f)
                    except EOFError:
                        break
                f = open('iteration_out.pk', 'wb') #check what is wb
                pickle.dump(d_old, f)
                pickle.dump(d, f)
                f.close()              
                d_old = copy.deepcopy(d)


            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg :
                if self.debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))

                if np.abs(fg - fp[i_min]) <= self.minfunc:
                    if (self.info == True): print('Stopping search: Swarm best objective change less than {:}'.format(self.minfunc))
                    #it_info.seek(0)
                    #it_info.write('-1')                
                    #it_info.close()              
                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= self.minstep:
                    if (self.info == True): print('Stopping search: Swarm best position change less than {:}'.format(self.minstep))
                    #it_info.seek(0)
                    #it_info.write('-2')                
                    #it_info.close()
                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]

            if self.debug:
                print 'Best after iteration {:}: {:} {:}'.format(it, g, fg)
            
            it_info.seek(0)
            it_info.write(str(it)) #update
            it += 1


        it_info.close()

        if (self.info == True):
            print('Stopping search: maximum iterations reached --> {:}'.format(self.maxiter))

        if not is_feasible(g):
            if (self.info == True): print("However, the optimization couldn't find a feasible design. Sorry")
        if self.particle_output:
            return g, fg, p, fp
        else:
	        return g, fg
