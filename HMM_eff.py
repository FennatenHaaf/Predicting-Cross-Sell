"""
This code aims to execute a Baum-Welch/forward-backward algorithm to estimate a Hidden Markov Model 
(with or without modelling the transition and initialisation probabilities by covariates) 


Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from time import perf_counter
#import numdifftools as nd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from scipy.stats import t

import extra_functions_HMM_eff as ef
import utils
import dataInsight


class HMM_eff:
    
    def __init__( self, outdir, outname, list_dataframes, reg_term,
                  max_method, list_dep_var, list_covariates = [], 
                  covariates = False, iterprint = False,
                  initparam = None, do_backup_folder = True, visualize_data = True):
        """
        Parameters
        ----------
        list_dataframes : list
            list consisting of the timeperiod-specific dataframes
        list_dep_var : list
            list consisting of all the names of the variables we use as dependent variables.
        list_covariates : boolean
            list consisting of all the names of the variables we use as covariates        
        covariates : boolean
            boolean indicating whether transition/state probabilities are modelled as logit model
        iterprint : boolean
            boolean indicating whether function evaluations within M-step are printed
        initparam : 

        """
        """Function for the Initialisation of a HMM object"""
           
        self.outdir = outdir
        self.outname = outname
        self.list_dataframes = list_dataframes
        self.list_dep_var = list_dep_var
        self.list_covariates = list_covariates
        self.initparam = initparam
        self.do_backup_folder = do_backup_folder
        self.visualize_data = visualize_data
        
        
        self.reg_term = reg_term
        self.max_method = max_method
        self.n_covariates = len(list_covariates) #initialise the number of covariates
        self.n_customers = self.list_dataframes[0].shape[0] #initialise the number of customers
        self.n_products = len(list_dep_var) #initialise the number of product
        self.T = len(list_dataframes) #initialise the number of dataframes, thus the timeperiod
        self.maximization_iters = 0
        self.covariates = covariates #initialise whether covariates are used to model the transition/state probabilities

        self.iterprint = iterprint #Iterprint True or False, will print x and iterations within a M-step

        self.function_for_hessian = None

        #compute per dependent variable the number of categories (possible values)
        self.n_categories = np.zeros((self.n_products))
        for i in range(0,self.T):
            for j in range(0,self.n_products):
                self.list_dataframes[i][list_dep_var[j]] = self.list_dataframes[i][list_dep_var[j]].fillna(0)
                n_per_df = self.list_dataframes[i][list_dep_var[j]].nunique(); #retrive the number of categories per product, per dataframe
                if n_per_df > self.n_categories[j]: #if number of categories is more than previously seen in other dataframes, update number of categories
                    self.n_categories[j] = n_per_df
        self.n_categories = self.n_categories.astype(int)
                    
        #Compute lists consisting of Y (dependent) and Z (covariates) matrices per timeperiod
        self.list_Y =[]

        if covariates == True:
            self.list_Z = []

        for i in range(0,self.T):
            if covariates == True:
                Z = list_dataframes[i][list_covariates]
                self.list_Z.append(Z.to_numpy())
            Y = list_dataframes[i][list_dep_var]
            self.list_Y.append(Y.to_numpy())
        
        # #Short method for creating categories, Z and Y
        # idx = pd.IndexSlice
        # self.n_categories2 = data_frame_collection.loc[idx[:], idx[:, list_dep_var]].nunique().unstack().max().to_numpy(
        # dtype='uint8')
        # self.list_Z2 = np.split(data_frame_collection.loc[idx[:], idx[:, list_covariates]].sort_index(axis=1).to_numpy(
        #     dtype='uint8'), 3, axis=1)
        # self.list_Y2 = np.split(data_frame_collection.loc[idx[:], idx[:, list_dep_var]].sort_index(axis=1).to_numpy(
        #     dtype='uint8'), 3,axis=1)
          
    def EM(self, n_segments, reg_term = 0.1, tolerance = 10**(-3), max_method = "BFGS", 
           random_starting_points = False, seed = None, bounded = None):
        """

        Parameters
        ----------
        n_segments : int
            number of segments to use for the estimation of the HMM
        tolerance : float
            convergence tolerance.
        max_method : string
            maximization method to use for the maximization step
        random_starting_points : boolean
            boolean indicating whether random starting points are used for the EM algorithm
        seed : int
            if random starting points are used, one can pass the seed
        bounded : tuple
            tuple consisting of the bound for the parameters, if one wants to use bounds

        Returns
        -------
        param_out : 1D array
            estimated parameters from the EM algorithm
        alpha_out : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}]
        beta_out  : 3D array
            estimated probabilities P[Y_{t+1:T} | X_{t}]
        shapes    : 2D array
            array consisting of shape and size of every single parameters matrix
        hes/hes_inv : 2D array
        """
        """function for running the EM algorithm"""
        
        #-----------------INITIALISE STARTING VALUES-------------------------
        
        if self.covariates == True:         #initialise parameters for HMM with the probabilities as logit model
        
            if random_starting_points == False: #initialise parameters with set startingvalues
                gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) ) #parameters for P(S_0 = s|Z)
                gamma_sr_0 =  np.ones( (n_segments-1,n_segments) ) #parameters for P(S_t = s | S_t-1 = r)
                gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) )  #parameters for P(S_t = s | S_t-1 = r)
                
                beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)
                for s in range(n_segments):
                    for p in range(0,self.n_products):
                        beta[s,p,0:self.n_categories[p]-1] = np.ones((1,self.n_categories[p]-1))                    
            
            else: 
                if seed == None: #initialise parameters with random startingvalues, without setting the seed
                    gamma_0 = np.random.uniform(low=-10, high=10, size=(n_segments-1, self.n_covariates+1)) #parameters for P(S_0 = s|Z)
                    gamma_sr_0 = np.random.uniform(low=-10, high=10, size=(n_segments-1,n_segments)) #parameters for P(S_t = s | S_t-1 = r)
                    gamma_sk_t = np.random.uniform(low=-10, high=10, size=(n_segments-1,self.n_covariates)) #parameters for P(S_t = s | S_t-1 = r)
        
                    beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)
                    for s in range(n_segments):
                        for p in range(0,self.n_products):
                            beta[s,p,0:self.n_categories[p]-1] = np.random.uniform(low=-5, high=5, size=(1,self.n_categories[p]-1)) 
                            
                else: #initialise parameters with random startingvalues, with a set seed
                    fixed_random_seed = np.random.RandomState(seed)
                    gamma_0 = fixed_random_seed.uniform(low=-10, high=10, size=(n_segments-1, self.n_covariates+1)) #parameters for P(S_0 = s|Z)
                    gamma_sr_0 = fixed_random_seed.uniform(low=-10, high=10, size=(n_segments-1,n_segments)) #parameters for P(S_t = s | S_t-1 = r)
                    gamma_sk_t = fixed_random_seed.uniform(low=-10, high=10, size=(n_segments-1,self.n_covariates)) #parameters for P(S_t = s | S_t-1 = r)
    
                    beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)
                    for s in range(n_segments):
                        for p in range(0,self.n_products):
                            beta[s,p,0:self.n_categories[p]-1] = fixed_random_seed.uniform(low=-5, high=5, size=(1,self.n_categories[p]-1))  
                            
            #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
            shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
            param = ef.param_matrices_to_list(self, n_segments, gamma_0 = gamma_0, gamma_sr_0 = gamma_sr_0, gamma_sk_t = gamma_sk_t, beta = beta)  #convert parametermatrices to list
            
            param_out = param #set name of parameterlist for the input of the algorithm
        
        else:         #initialise parameters for HMM without the probabilities as logit model
            A = 1/n_segments * np.ones((n_segments-1,n_segments)) #parameters of P(S_t = s | S_t-1 = r)
            pi = 1/n_segments * np.ones((n_segments-1))  #parameters for P(S_0 = s)
            
            b = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) ##parameters for P(Y| S_t = s)
            for s in range(n_segments):
                for p in range(0,self.n_products):
                    b[s,p,0:self.n_categories[p]-1] = 0.5 * np.ones((1,self.n_categories[p]-1)) 
            
            #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
            shapes = np.array([[A.shape,A.size], [pi.shape, pi.size], [b.shape, b.size]], dtype = object)
            param = ef.param_matrices_to_list(self, n_segments, A = A, pi = pi, b = b) #convert parametermatrices to list
            param_out = param #set name of parameterlist for the input of the algorithm

            
        # If initial parameters had already been specified,
        # Replace what we just initialised (but we can still use the shapes)
        if not( isinstance(self.initparam, type(None)) ): 
            param_out = self.initparam
        
            
        print(f"Starting values: {utils.printarray(param_out)}") #print starting values
        #print(f"Starting values: {param}") #print starting values

        #initialise
        self.iteration = 0
        difference = True

        if self.do_backup_folder:
            self.starting_datetime = utils.get_datetime()
        
        print(f"****Starting EM prodecure, at {utils.get_time()}****")
        print(f"tolerance: {tolerance}")
        print(f"number of parameters: {len(param_out)}")
        print(f"maximization method: {max_method}")
        print(f"random starting points: {random_starting_points}")

        alpha_out = np.zeros((n_segments, self.n_customers, self.T))
        beta_out = np.zeros((n_segments, self.n_customers, self.T))
        logl_out = 0
        start_EM = utils.get_time()
            
        #----------------------Start EM procedure-------------------------
        while difference:
                
            #update parameters
            param_in = param_out 
            alpha_in = alpha_out
            beta_in = beta_out
            logl_in = logl_out
            
            #perform forward-backward procedure (expectation step of EM) 
            start = utils.get_time()
            alpha_out, beta_out = self.forward_backward_procedure(param_in, shapes, n_segments)
            start1 = utils.get_time()
            print(f"E-step duration: {utils.get_time_diff(start,start1)} ")


            #perform maximisation step 
            param_out = self.maximization_step(alpha_out, beta_out, param_in, shapes, n_segments, reg_term, max_method, bounded)
            logl_out = self.loglikelihood(param_out, shapes, n_segments)

            end = utils.get_time()#set end time to time maximisation step
            diff = utils.get_time_diff(start,end)#get difference of start and end time, thus time to run maximisation 
            print(f"Finished iteration {self.iteration}, duration M step {diff}")

            #hes = nd.Hessian(self.loglikelihood)(param_out,  shapes, n_segments)
            #print(f"Hessian: {hes}")
            
            if self.covariates:
                gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(self, n_segments, param_out, shapes)
                print(f"{param_out}")
            
            # Save the output to a text file
            with open(f'{self.outdir}/{self.outname}.txt', 'w') as f:
                
                f.write(f"time: {utils.get_time()} \n")
                f.write(f"iteration: {self.iteration} \n\n")
                
                f.write(f"dependent variable: {self.list_dep_var} \n")
                f.write(f"covariates: {self.list_covariates} \n")   
                f.write(f"number of covariates: {len(self.list_covariates)} \n\n")   
    
                f.write(f"number of parameters: {len(param_out)}\n")
                f.write(f"regularisation term: {reg_term}\n")
                f.write(f"tolerance: {tolerance}\n")
                f.write(f"maximization method: {max_method}\n")
                f.write(f"random starting points: {random_starting_points}\n\n")

                arraystring = utils.printarray(param_out)
                paramstring = f"param_out = np.array({arraystring}) \n\n"
                f.write(paramstring)
                
                f.write(f"LogLikelihood value: {logl_out}")

            #Backup files into backup folder
            if self.do_backup_folder:
                utils.create_result_archive(self.outdir, archive_name = "hmm_iterations",subarchive_addition =
                                    self.starting_datetime, files_string_to_archive_list = ['crosssell'],
                                            file_string_to_exclude_list = ['_HESSIAN'] )

            #compute difference to check convergence 
            if self.iteration != 0:
               difference = (np.linalg.norm(abs(alpha_in - alpha_out)) > tolerance) & (np.linalg.norm(abs(beta_in - beta_out)) > tolerance)
               print(f"norm absolute difference alpha: {np.linalg.norm(abs(alpha_in - alpha_out))}")
               print(f"norm absolute difference beta: {np.linalg.norm(abs(beta_in - beta_out))}")

               #difference = (np.max(abs(param_in-param_out)) > tolerance) #set difference of input and output of model-parameters
               #print(f"max difference: {np.max(abs(param_in-param_out))}")

               #difference = abs(logl_out - logl_in) > tolerance
               print(f"LogLikelihood value: {logl_out}")
               print(f"Difference LogLikelihood value: {logl_out - logl_in}")

            #if statement to set a breakpoint if desired
            if self.iteration == 1:
                print('breakpoint')
                
            if self.iteration == 20:
                print('breakpoint')
                
            self.iteration = self.iteration + 1 #update iteration
        
        #compute and print time for the complete EM algorithm
        end_EM = utils.get_time()
        diffEM = utils.get_time_diff(start_EM,end_EM)
        print(f"Total EM duration: {diffEM}")
        
        #-----------------------calculate hessian------------------------------
        
        print(f"Calculating Hessian at {utils.get_time()}")
        # hes = nd.Hessian(self.loglikelihood)(param_out,  shapes, n_segments)
        # print(f"Done calculating at {utils.get_time()}!")
        
        print(f"Doing another BFGS step for the covariance at {utils.get_time()}")
        #do one last minimisation of the loglikelihood itself to retrieve the hessian 
        hess_inv, dfSE, param_afterBFGS = self.get_standard_errors(param_out, n_segments)
        # param_out, hess_inv = self.maximization_step(alpha_out, beta_out, param_in, 
        #                                              shapes, n_segments, reg_term,
        #                                              max_method, bounded, end = True)
        logl_outafterBFGS = self.loglikelihood(param_afterBFGS, shapes, n_segments)
        
        print(f"Done calculating at {utils.get_time()}!")
              
        if self.do_backup_folder:
            utils.create_result_archive(self.outdir, archive_name = "hmm_iterations", subarchive_addition =
            self.starting_datetime, files_string_to_archive_list = ['_HESSIAN'])

    
        return param_afterBFGS, alpha_out, beta_out, shapes, hess_inv #, hes
     
        
     
     
    def forward_backward_procedure(self, param, shapes, n_segments, data = None):
        """

        Parameters
        ----------
        param  : 1D array
            estimated parameters from the previous M-step
        shapes : 2D array
            array consisting of shape and size of every single parameter matrix
        n_segments : int
            number of segments being used for the estimation of the HMM
        data : list of dataframes
            dataset from which one wants to compute alpha and beta, has to be specified if one wants to get results from customers outside the training set
        Returns
        -------
        alpha_return : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}]
        beta_return  : 3D array
            estimated probabilities P[Y_{t+1:T} | X_{t}]
        """
        """function for the E-step (forward and backward procedure)"""
        
        #get probabilites P[Y_{itp} = c | X_{it} = s]
        p_js = ef.prob_p_js(self, param, shapes, n_segments)
        
        #initialise matrices for the return
        alpha_return = np.zeros((n_segments, self.n_customers, self.T)) 
        beta_return = np.zeros((n_segments, self.n_customers, self.T))
        
        if data == None:
            n_cust = self.n_customers
            T = self.T
        else:
            n_cust = len(data[0])
            T = len(data)
            
        for i in range(0,n_cust):
            for t in range(0,T):
                v = self.T - t - 1
                
                #get the data from the right person at the right time
                if data == None: # if data is used from the trainingset
                    Y_t = np.array([self.list_Y[t][i,:]])
                    if self.covariates == True:
                        Z_t = np.array([self.list_Z[t][i,:]])
                    else:
                        Z_t = []
                else: # if data is used from a 'new' customer
                    data_t = data[t]
                    Y_t = np.array([ data[self.list_dep_var][t][i,:] ])
                    if self.covariates == True:
                        Z_t = np.array([ data[self.list_covariates][t][i,:] ])
                    
                #compute alpha and beta if t = 0
                if t == 0:
                    P_y_given_s_t = ef.prob_P_y_given_s(self, Y_t, p_js, n_segments)
                    P_s_given_Z_t = ef.prob_P_s_given_Z(self, param, shapes, Z_t, n_segments)

                    alpha_return[:,i,t] = np.multiply(P_y_given_s_t, P_s_given_Z_t).flatten()
                    beta_return[:,i,v] = np.ones((n_segments))
                #compute alpha and beta if t >= 1
                else:
                    #get the data from the right person at the right time, data_v1 is for the backward algorithm
                    Y_v1 = np.array([self.list_Y[v+1][i,:]])
                    if self.covariates == True:
                        Z_v1 = np.array([self.list_Z[v+1][i,:]])
                    else:
                        Z_v1 = []
                        
                    #compute probabilties for estimating alpha and beta
                    P_s_given_r_t = ef.prob_P_s_given_r(self, param, shapes, Z_t, n_segments)
                    P_s_given_r_v1 = ef.prob_P_s_given_r(self, param, shapes, Z_v1, n_segments)
                    
                    P_y_given_s_t = ef.prob_P_y_given_s(self, Y_t, p_js, n_segments)
                    P_y_given_s_v1 = ef.prob_P_y_given_s(self, Y_v1, p_js, n_segments)

                    #before going into the for-loop for computing them, initialise sums 
                    sum_alpha = np.zeros( (n_segments) )
                    sum_beta = np.zeros( (n_segments) )
                    #for loop for computing the sum
                    for r in range(0,n_segments):
                            sum_alpha = sum_alpha + alpha_return[r,i,t-1] * np.multiply(P_s_given_r_t[:,:,r], P_y_given_s_t.flatten())
                            sum_beta = sum_beta + beta_return[r,i,v+1] * np.multiply(P_s_given_r_v1[:,r,:], P_y_given_s_v1[:,r])
                            
                    #add computed sums to alpha and beta matrices
                    alpha_return[:,i,t] = sum_alpha
                    beta_return[:,i,v]  = sum_beta
                
        return alpha_return, beta_return
      
    def maximization_step(self, alpha, beta, param_in, shapes, n_segments, reg_term, max_method, bounded, end = False):
        """
        Parameters
        ----------
        alpha : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}], coming from the E-step (forward and backward procedure)
        beta  : 3D array
            estimated probabilities P[Y_{t+1:T} | X_{t}], coming from the E-step (forward and backward procedure)
        param_in : 1D array
            parameters estimated at the previous M_step
        shapes : 2D array
            array consisting of shape and size of every single parameter matrix
        n_segments : int
            number of segments being used for the estimation of the HMM
        reg_term : float
            regularisation term, indicates the penalty for large parameters
        max_method : string
            maximization method to use for the maximization step
        bounded : tuple
            indicates bounds for the parameters, if desired
        end : boolean
            indicates whether it is the last iteration, and the hessian inverse has also to be returned
        Returns
        -------
        param_out : 1D array
            estimated parameters in current M-step
        hess_inv : 2D array
            estimated inverse of hessian
        """
        """function for the maximization step"""
        
        #calculate multiple probabilities which are constant in the function that has to be minimized
        P_s_given_Y_Z = ef.state_event(self, alpha, beta)
        p_js_cons = ef.prob_p_js(self, param_in, shapes, n_segments)
        P_s_given_Y_Z_ut = np.multiply(alpha, beta)
        
        list_P_s_given_r = []
        list_P_y_given_s = []
        
        for t in range(0,self.T):
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]   
            else: 
                Z = []
            P_s_given_r = ef.prob_P_s_given_r(self, param_in, shapes, Z, n_segments)
            P_y_given_s = ef.prob_P_y_given_s(self, Y, p_js_cons, n_segments)
            list_P_s_given_r.append(P_s_given_r)
            list_P_y_given_s.append(P_y_given_s)
            
        #set startingvalues
        x0 = param_in
            
        self.maximization_iters = 0

        #set options for the different optimization routines
        #fatol_value = 1e-3 + (1e-1)/np.exp( ( self.iteration / 10) )
        #xatol_value = 1e-1 + (1 - 1e-1)/np.exp( ( self.iteration / 100) )
        #max_iter_value = 2.5*10**4
        # print('fatol: ', fatol_value, ' and xatol :', xatol_value )
        #minimize_options = {'disp': True, 'fatol': fatol_value, 'xatol': xatol_value, 'maxiter': max_iter_value}
        # minimize_options_NM = {'disp': True, 'adaptive': False, 'xatol': 1, 'fatol': 0.1}
        minimize_options_NM = {'disp': True, 'adaptive': False, 'xatol': 1e-1, 'fatol': 1e-1}
        minimize_options_BFGS = {'disp': True, 'maxiter': 99999} 
    
        #run the minimisation
        if (max_method == 'Nelder-Mead') & (end == False): #if Nelder-Mead is used and it is not the last maximisation step
            if self.iteration <= 9999999: #the first X iterations Nelder-Mead is used, thereafter BFGS
                param_out = minimize(self.optimization_function, x0, args=(alpha, beta, shapes,
                                         n_segments, reg_term, P_s_given_Y_Z, list_P_s_given_r, list_P_y_given_s, p_js_cons, P_s_given_Y_Z_ut),
                                     method=max_method,options= minimize_options_NM)
                #param_out = minimize(self.loglikelihood, x0, args=(shapes, n_segments),
                 #                    method=max_method,options= minimize_options_NM)


                ##EXTRACT SIMPLEX VALUES
                # final_xs = param_out.final_simplex[0]
                # final_fs = param_out.final_simplex[1]

                # ##PRINT ITERATION DIFFERENCES BETWEEN SIMPLEX ITERS
                # if final_xs.shape[0] > 1:
                #     for j, x in reversed(list(enumerate(final_xs[:-1]))):
                #         i = j + 1
                #         print(f"For simplex iteration number {final_xs.shape[0] - j} :")
                #         print(f"Log difference is {final_fs[j] - final_fs[i]:.4f} with previous logl {float(final_fs[i]):.5f} and "
                #               f"current "
                #               f"logl {float(final_fs[j]):.5f}")
                #         xdiff = np.abs(final_xs[j] - final_xs[i])
                #         x_max_loc = np.argmax(xdiff)
                #         print(f"Maximal x difference between simplex iterations is {float(xdiff[x_max_loc]):.4f} at position"
                #               f" {x_max_loc} "
                #               f"with current x: "
                #               f"{float(final_xs[j][x_max_loc])}:.5f and with previous x : {float(final_xs[i][x_max_loc])}:.5f")
                # else:
                #     print(f"Only one iteration")

                ###SHOW LARGEST DIFFERENCE FOR STOPPING CRITERION
                # nthlargest = 10
                # if final_xs.shape[0] >= (nthlargest + 1):
                #     with np.printoptions(threshold = np.inf):
                #         print(f"\n 10 largest Differences in logl according to simplex before termination"
                #               f": \n {np.sort(np.ravel(np.abs(final_fs[2:] - final_fs[1])))[-nthlargest:] }")
                #         print(f"\n 10 largest Differences in logl according to simplex at termination"
                #               f": \n {np.sort(np.ravel(np.abs(final_fs[1:] - final_fs[0])))[-nthlargest:] }")
                #         print(f"\n 10 largest Differences in x according to simplex before termination"
                #               f": \n {np.sort(np.ravel(np.abs(final_xs[2:] - final_xs[1])))[-nthlargest:] }")
                #         print(f"\n 10 largest Differences in x according to simplex at termination"
                #               f": \n {np.sort(np.ravel(np.abs(final_xs[1:] - final_xs[0])))[-nthlargest:] }")
                # else:
                #     print("Not enough simplex iterations to see largest differences according to method")




            else:
                param_out = minimize(self.optimization_function, x0, args=(alpha, beta, shapes,
                                         n_segments, reg_term, P_s_given_Y_Z, list_P_s_given_r, list_P_y_given_s, p_js_cons, P_s_given_Y_Z_ut),
                                         method='BFGS',options= minimize_options_BFGS)
                #param_out = minimize(self.loglikelihood, x0, args=(shapes, n_segments),
                #                      method='BFGS',options= minimize_options_BFGS)
            return param_out.x

        elif (max_method == 'BFGS') & (end == False): #if BFGS is used and it is not the last maximisation step
            if bounded == None: #if parameters are not bounded, use BFGS, otherwise use L-BFGS-B
                #param_out = minimize(self.optimization_function, x0, args=(alpha, beta, shapes,
                 #                  n_segments, reg_term, P_s_given_Y_Z, list_P_s_given_r, list_P_y_given_s, p_js_cons, P_s_given_Y_Z_ut),
                 #                  method='BFGS',options= minimize_options_BFGS)
                param_out = minimize(self.loglikelihood, x0, args=(shapes, n_segments),
                                     method='BFGS',options= minimize_options_BFGS)
            else: 
                ub = bounded[1] 
                lb = bounded[0]
                bnds = ((lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),
                            (lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),(lb,ub),
                            (lb,ub))
                param_out = minimize(self.optimization_function, x0, args=(alpha, beta, shapes,
                                      n_segments, reg_term, P_s_given_Y_Z, list_P_s_given_r, list_P_y_given_s, p_js_cons, P_s_given_Y_Z_ut),
                                      method='L-BFGS-B',options= minimize_options_BFGS, bounds = bnds)
                #param_out = minimize(self.loglikelihood, x0, args=(shapes, n_segments),
                #                      method='L-BFGS-B',options= minimize_options_BFGS, bounds = bnds)
            return param_out.x
        elif end == True: #if it is the last iteration, minimize the loglikelihood itself (instead of expected complete data loglikelihood)
            param_out = minimize(self.loglikelihood, x0, args=(shapes, n_segments),
                                 method='BFGS',options= minimize_options_BFGS)
            self.param_out = param_out
            return param_out.x, param_out.hess_inv
        
    
    def optimization_function(self, x, alpha, beta, shapes, n_segments, reg_term, 
                              P_s_given_Y_Z, list_P_s_given_r, list_P_y_given_s, p_js, P_s_given_Y_Z_ut):
        """
        Parameters
        ----------
        x     : 1D array
            parameters over which the maximisation must be done
        alpha : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}], coming from the E-step (forward and backward procedure)
        beta  : 3D array
            estimated probabilities P[Y_{t+1:T} | X_{t}], coming from the E-step (forward and backward procedure)
        shapes : 2D array
            array consisting of shape and size of every single parameter matrix
        n_segments : int
            number of segments being used for the estimation of the HMM
        P_s_given_Y_Z : 3D array
            probabilties P[X_{it} = s | Y_{it}, Z_{it}]
        list_P_s_given_r : list 
            probabilties P[X_{it} = s | X_{it} = r, Z_{it}]
        list_P_y_given_s : list
            probabilties P[Y_{it} = c | X_{it} = s]
        p_js : 3D array
            probabilties P[Y_{itp} = c | X_{it} = s]
        P_s_given_Y_Z_ut : 3D array
            element-wise multiplication of alpha and beta
        Returns
        -------
        param_out : float
            value of the complete data loglikelihood

        """
        """function for the optimization function"""                    
        
        #compute probabilties P[Y_{itp} = c | X_{it} = s], which is person and time-invariant
        p_js_max = ef.prob_p_js(self, x, shapes, n_segments)

        #set complete data loglikelihood to zero
        logl = 0;

        Y = self.list_Y[0]
        if self.covariates == True:
            Z = self.list_Z[0]
        else: 
            Z = np.array([])
            
        P_s_given_Y_Z_0 = np.transpose(P_s_given_Y_Z[:,:,0]) #s x i x t
        
        #t=0, term 1
        P_s_given_Z = ef.prob_P_s_given_Z(self, x, shapes, Z, n_segments)  #i x s
        mult = np.multiply(P_s_given_Y_Z_0, np.log(P_s_given_Z + 10**(-300)))
        logl += np.sum(mult)
        
        #t=0, term 3
        P_y_given_s_0 = ef.prob_P_y_given_s(self, Y, p_js_max, n_segments)#ixs
        mult = np.multiply(P_s_given_Y_Z_0, np.log(P_y_given_s_0 + 10**(-300)))
        logl += np.sum(mult)

        for t in range(1,self.T):
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]   
            else: 
                Z = []
    
            #t=t, term 2
            #for r in range(0,n_segments):
            # These do not depend on the segment - use as input for joint event function
            P_s_given_r_cons = list_P_s_given_r[t]
            P_y_given_s_cons = list_P_y_given_s[t]
            
            P_s_given_r_max = ef.prob_P_s_given_r(self, x, shapes, Z, n_segments)
            P_sr_given_Y_Z = ef.joint_event(self, alpha, beta, t, n_segments,
                                                P_s_given_Y_Z_ut, P_s_given_r_cons, P_y_given_s_cons)
            mult = np.multiply(P_sr_given_Y_Z, np.log(P_s_given_r_max + 10**(-300)))
            logl += np.sum(mult)
            
            #t=t, term 3
            P_y_given_s_max = ef.prob_P_y_given_s(self, Y, p_js_max, n_segments) #ixs
            P_s_given_Y_Z_t = np.transpose(P_s_given_Y_Z[:,:,t]) #ixs
            mult = np.multiply(P_s_given_Y_Z_t, np.log(P_y_given_s_max + 10**(-300)))
            logl += np.sum(mult)
        
        
        logl = -logl + np.sum(abs(x)) * reg_term
        self.maximization_iters += 1
        if self.iterprint:
            if (self.maximization_iters % 1000 == 0):  # print alleen elke 1000 iterations
               print('function value:', logl,' at iteration ',self.maximization_iters)

        return logl


    
    
    def loglikelihood(self, param, shapes, n_segments):
        """
        Parameters
        ----------
        param     : 1D array
            estimated parameters from the loglikelihood
        shapes : 2D array
            array consisting of shape and size of every single parameter matrix
        n_segments : int
            number of segments being used for the estimation of the HMM
        Returns
        -------
        param_out : float
            value of the loglikelihood

        """
        """function for calculating the loglikelihood""" 
        
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(self, n_segments, param, shapes)
        
        p_js = ef.prob_p_js(self, param, shapes, n_segments)
        logl = 0
        logl_i = np.zeros(self.n_customers)
        
                    
        for t in range(0,self.T):
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]
            else:
                Z = []
                        
            if t == 0:
                P_s_given_Z = ef.prob_P_s_given_Z(self, param, shapes, Z, n_segments) #[ixs]
                P_s_given_Z = P_s_given_Z[:,np.newaxis,:]
                
                P_Y_given_S = ef.prob_P_y_given_s(self, Y, p_js, n_segments)
                P_Y_given_S = np.eye(n_segments) * P_Y_given_S[:,np.newaxis,:]

                likelihood = np.matmul(P_s_given_Z, P_Y_given_S)
                    
            elif t == (self.T - 1):
                P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments) #[sxs]
                P_s_given_r = P_s_given_r.swapaxes(1,2)

                P_Y_given_S = ef.prob_P_y_given_s(self, Y, p_js, n_segments) 
                P_Y_given_S = P_Y_given_S[:,:,np.newaxis]

                
                mat = np.matmul(P_s_given_r, P_Y_given_S) 
                likelihood = np.matmul(likelihood, mat)
            else: 
                P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments) #[sxs]
                P_s_given_r = P_s_given_r.swapaxes(1,2)
                
                P_Y_given_S = ef.prob_P_y_given_s(self, Y, p_js, n_segments) 
                P_Y_given_S = np.eye(n_segments) * P_Y_given_S[:,np.newaxis,:]

                mat = np.matmul(P_s_given_r, P_Y_given_S)
                likelihood = np.matmul(likelihood, mat)
                    
        logl_i = np.log(likelihood + 10**(-300))
            
        logl = - np.sum(logl_i) + np.sum(abs(param)) * self.reg_term
        
        return logl

            

# =============================================================================
#  METHODS TO PROCESS OUTPUT       
# =============================================================================

    def predict_product_ownership(self, param, shapes, n_segments, alpha):
        """
        Parameters
        ----------
        param     : 1D array
            estimated parameters of the HMM
        shapes : 2D array
            array consisting of shape and size of every single parameter matrix
        n_segments : int
            number of segments being used for the estimation of the HMM
        alpha : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}], coming from the E-step (forward and backward procedure) from the HMM
  
        Returns
        -------
        prediction : 3D array
            prediction of the amount (third index) every customer (first index) owns of a product (second index)

        """
        """function for predicting the amount every customer owns of a product""" 
        
        if self.covariates == True:
            Z = self.list_Z[self.T-1]

            gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(self, n_segments, param, shapes)

            p_js = ef.prob_p_js(self, param, shapes, n_segments) #s x p x c
            P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments)
            nextstate = np.zeros((self.n_customers, n_segments)) #i x s
            prediction = np.zeros((self.n_customers, self.n_products, max(self.n_categories)))


            for i in range(0,self.n_customers):

                probstate = alpha[:, i, self.T-1] / np.sum(alpha[:, i, self.T-1])

                for s in  range(0,n_segments):
                    nextstate[i,:] += probstate[s] * P_s_given_r[i,:,s]

                    #i x s    s x p x c
                for p in range(0,self.n_products):
                    for c in range(0,self.n_categories[p]):
                        pred = 0 
                        for s in range(0,n_segments):
                            pred += nextstate[i,s]* p_js[s,p,c]
                        prediction[i,p,c] = pred


            return prediction

    def active_value(self, param, n_segments, t, data = None):
        """
        Parameters
        ----------
        param     : 1D array
            estimated parameters of the HMM
        n_segments : int
            number of segments being used for the estimation of the HMM
        t : int
            time for which one wants an active value
        alpha : 3D array
            estimated probabilities P[Y_{0:t}, X_{t}], coming from the E-step (forward and backward procedure) from the HMM
        data : list of dataframes
            dataset from which one wants to compute the active value, has to be specified if one wants to get results from customers outside the training set
        Returns
        -------
        active_value : 1D array
            active value for every customer
        """
        """function that computes an active value for every customer by means of the estimated parameters""" 

        #----------- Initialise everything so that we get the shapes-------------
        gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
        gamma_sr_0 =  np.ones( (n_segments-1,n_segments) )
        gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) ) 
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1))
        # for s in range(n_segments):
        #     for p in range(0,self.n_products):
        #         beta[s,p,0:self.n_categories[p]-1] = 10*np.ones((1,self.n_categories[p]-1)) 
                
        #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
        shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], 
                           [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
              

        alpha, beta = self.forward_backward_procedure(param, shapes, n_segments, data = data)

        P_s_given_Y_Z = ef.state_event(self, alpha, beta)
        active_value = np.argmax(P_s_given_Y_Z, axis = 0)
        active_value_t = active_value[:, t-1]
        
        return active_value_t


    def cross_sell_yes_no(self, param, n_segments, active_value, data = None, tresholds = [0.5,0.8], order_active_high_to_low = [0,1,2]):
        """
        Parameters
        ----------
        param     : 1D array
            estimated parameters of the HMM
        n_segments : int
            number of segments being used for the estimation of the HMM
        active value : 1D array
            active value of every customer
        data : list of dataframes
            dataset from which one wants to compute whether a cross sell is possible, has to be specified if one wants to get the result from customers outside the training set
        tresholds : list
            tresholds that indicate when a customers is eligible for a cross sell
        order_active_high_to_low : list
            because in the HMM it is not specified which segment represents the level of activeness, this list does
        Returns
        ------- 
        dif_exp_own : 2D array
            array representing for every customer (row) for every product (column) the difference between the expected ownership and the real ownership
        cross_sell_target : 2D array
            array representing whether a customers (row) is eligible for cross sell targeting regarding a certain product (columns)
        cross_sell_self : 2D array
            array representing whether a customers (row) cross sells a certain product (columns) itself
        cross_sell_total : 2D array
            array representing both cross_sell_target and cross_sell_self
        """
        """function that gives whether a customer is eligible for a cross sell""" 
        

       #----------- Initialise everything so that we get the shapes-------------
        gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
        gamma_sr_0 =  np.ones( (n_segments-1,n_segments) )
        gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) ) 
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1))
        # for s in range(n_segments):
        #     for p in range(0,self.n_products):
        #         beta[s,p,0:self.n_categories[p]-1] = 10*np.ones((1,self.n_categories[p]-1)) 
                
        #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
        shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], 
                           [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)

        if data == None:
            alpha, beta = self.forward_backward_procedure(param, shapes, n_segments)
            Y = self.list_Y[self.T-1]
        else:
            alpha, beta = self.forward_backward_procedure(param, shapes, n_segments, data)
            Y = np.array([ data[self.list_dep_var][len(data)-1] ])


        prod_own = self.predict_product_ownership(param, shapes, n_segments, alpha)
        
        expected_n_prod = np.zeros((self.n_customers, self.n_products))
        dif_exp_own = np.zeros((self.n_customers, self.n_products))
        cross_sell_target = np.zeros((self.n_customers, self.n_products))
        cross_sell_self = np.zeros((self.n_customers, self.n_products))
        cross_sell_total = np.zeros((self.n_customers, self.n_products))
        

        for i in range(0, self.n_customers):
            for p in range(0,self.n_products):
                for c in range(0,self.n_categories[p]):
                    expected_n_prod[i,p] = expected_n_prod[i,p] + c*prod_own[i,p,c]
                    
                dif_exp_own[i,p] = expected_n_prod[i,p] - Y[i,p]
                if dif_exp_own[i,p] >= tresholds[1]:
                    if active_value[i] == order_active_high_to_low[0]:
                        cross_sell_target[i,p] = False
                        cross_sell_self[i,p] = True
                        cross_sell_total[i,p] = True
                    if active_value[i] == order_active_high_to_low[1]:
                        cross_sell_target[i,p] = True
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = True               
                    if active_value[i] == order_active_high_to_low[2]:
                        cross_sell_target[i,p] = True
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = True  
                elif (dif_exp_own[i,p] < tresholds[1]) & (dif_exp_own[i,p] >= tresholds[0]):
                    if active_value[i] == order_active_high_to_low[0]:
                        cross_sell_target[i,p] = True
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = True
                    if active_value[i] == order_active_high_to_low[1]:
                        cross_sell_target[i,p] = True
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = True               
                    if active_value[i] == order_active_high_to_low[2]:
                        cross_sell_target[i,p] = True
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = True 
                else:
                    if active_value[i] == order_active_high_to_low[0]:
                        cross_sell_target[i,p] = False
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = False
                    if active_value[i] == order_active_high_to_low[1]:
                        cross_sell_target[i,p] = False
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = False               
                    if active_value[i] == order_active_high_to_low[2]:
                        cross_sell_target[i,p] = False
                        cross_sell_self[i,p] = False
                        cross_sell_total[i,p] = False 
                        
        return dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own
    
    def number_of_cross_sells(self, cross_sell_target, cross_sell_self, cross_sell_total):
        """
        Parameters
        ----------
        cross_sell_target : 2D array
            array representing whether a customers (row) is eligible for cross sell targeting regarding a certain product (columns)
        cross_sell_self : 2D array
            array representing whether a customers (row) cross sells a certain product (columns) itself
        cross_sell_total : 2D array
            array representing both cross_sell_target and cross_sell_self
        Returns
        -------
        n_cross_sells : 2D array
            array representing the number of cross sells for every product (row) and target/self/total (column)
        """
        """function calculates the number of cross sell that are predicted in cross_sell_yes_no""" 
        
        # initialise return
        n_cross_sells = np.zeros((self.n_products, 3))
        
        # for every every product, calculat the number of cross sells
        for p in range(0,self.n_products):
                n_cross_sells[p,0] = np.count_nonzero(cross_sell_target[:,p])
                n_cross_sells[p,1] = np.count_nonzero(cross_sell_self[:,p])
                n_cross_sells[p,2] = np.count_nonzero(cross_sell_total[:,p])

        return n_cross_sells
    
    
    def interpret_parameters(self,parameters,n_segments, person_index = 0):
        """This function aims to visualise and interpret the results for
        the parameters of a specific person (standard: the first one)"""
        
        #----------- Initialise everything so that we get the shapes-------------
        gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
        gamma_sr_0 =  np.ones( (n_segments-1,n_segments) )
        gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) ) 
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1))
   
        #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
        shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], 
                           [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
                   
        
        #----------- Look at the actual parameters -------------
        print("Interpreting the following parameters")
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(self, n_segments, parameters, shapes)
        print(f"Gamma_0: {gamma_0}")
        print(f"Gamma_sr_0: {gamma_sr_0}")
        print(f"Gamma_sk_t: {gamma_sk_t}")
        print(f"Beta: {beta}")
        
        #perform ONE forward-backward procedure (expectation step of EM) 
        alpha, beta = self.forward_backward_procedure(parameters, shapes, n_segments)
        
        #Change gamma_sr_0 into a transition probabilities matrix and visualize
        transition_probs = ef.gamma_sr_0_to_trans(self, parameters, shapes, n_segments)
        #Now interpret the impacts of the covariates
        y_axis = np.arange(n_segments)
        x_axis = np.arange(n_segments)
        dataInsight.visualize_matrix(self,transition_probs, x_axis, y_axis,
                              "Segment","Segment", 
                              "Base transition probabilities")
        
        #Now visualise the logit coefficients for the covariates (gamma_sk_t)
        y_axis = np.arange(n_segments-1) # we have the last segment as base case (?)
        x_axis = self.list_covariates
        dataInsight.visualize_matrix(self,gamma_sk_t, x_axis, y_axis,
                              "Covariates", "Segments",
                              "Logit coefficients on segment membership probabilities",
                              diverging = True)
        
        
        
        # ------- Visualize the pjs ---------
        p_js = ef.prob_p_js(self, parameters, shapes, n_segments)
        for seg in range(0, n_segments):
            y_axis = self.list_dep_var
            x_axis = np.arange(np.max(self.n_categories))
            matrix = p_js[seg,:,:]
            dataInsight.visualize_matrix(self,matrix, x_axis, y_axis, "Level",
                                  "dependent variable",
                                   f"P_js for segment {seg}")
        
        # ------Visualize Ps_given_Y_Z---------
        P_s_given_Y_Z = ef.state_event(self, alpha, beta)
        y_axis = np.arange(n_segments)
        x_axis = np.arange(self.T)
        dataInsight.visualize_matrix(self,P_s_given_Y_Z[:,person_index,:], x_axis, y_axis,
                              "Time","Segment",
                              f"P_s_given_Y_Z for person {person_index}")
        
        # Visualize Ps_given_Y_Z_ut??
        P_s_given_Y_Z_ut = np.multiply(alpha, beta)
        
    
        # ------Visualize time dependent variables ---------
        list_P_s_given_r = []
        list_P_y_given_s = []
        list_P_s_given_Z = []
        for t in range(0,self.T):
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]   
            else: 
                Z = []
            P_s_given_r = ef.prob_P_s_given_r(self, parameters, shapes, Z, n_segments)
            P_y_given_s = ef.prob_P_y_given_s(self, Y, p_js, n_segments)
            P_s_given_Z = ef.prob_P_s_given_Z(self, parameters, shapes, Z, n_segments) #[ixs]
            list_P_s_given_r.append(P_s_given_r)
            list_P_y_given_s.append(P_y_given_s)
            list_P_s_given_Z.append(P_s_given_Z)
        
            
        # Visualize P_s_given_r for the last T
        title = f"P_s_given_r for person {person_index} at time T"
        y_axis = np.arange(n_segments)
        x_axis = np.arange(n_segments)
        dataInsight.visualize_matrix(self,P_s_given_r[person_index,:,:], x_axis, y_axis,
                              "segment","segment",title)
        
        # Visualize P_s_given_Z
        # title = f"P_s_given_r for person {person_index} at time T"
        # y_axis = np.arange(n_segments)
        # x_axis = np.arange(n_segments)
        # self.visualize_matrix(P_s_given_r[person_index,:,:], x_axis, y_axis, 
        #                       "segment","segment",title)
        self.function_for_hessian = lambda x, reg: self.optimization_function(x,alpha,beta,shapes,n_segments,reg,P_s_given_Y_Z,
                                                                         list_P_s_given_r,list_P_y_given_s,p_js, P_s_given_Y_Z_ut)

        return p_js, P_s_given_Y_Z, gamma_0, gamma_sr_0, gamma_sk_t, transition_probs
        
        
        
                
    def cross_sell_new_cust(self, data, param_cross, param_act, n_segments_cross, act_obj, t,
                            n_segments_act = 3, tresholds = [0.5, 0.8], order_active_high_to_low = [0,1,2]): 
        """
        Parameters
        ----------
        data : list of dataframes
            data of the customers for one wants to know whether a cross sell is possible
        param_cross    : 1D array
            estimated parameters of the HMM for cross sells
        param_act : 1D array
            estimated parameters of the HMM for the active value
        n_segments_cross : int
            number of segments being used for the estimation of the HMM for cross sells
        act_obj : object
            object of HMM_eff for estimating the active value
        t : int
            time on which one wants to know whether a cross sell is possible
        n_segments_act : int
            number of segments used for the HMM for the active value
        tresholds : list
            tresholds that indicate when a customers is eligible for a cross sell
        order_active_high_to_low : list
            because in the HMM it is not specified which segment represents the level of activeness, this list does
        Returns
        -------
        cross_sell_target : 2D array
            array representing whether a customers (row) is eligible for cross sell targeting regarding a certain product (columns)
        cross_sell_self : 2D array
            array representing whether a customers (row) cross sells a certain product (columns) itself
        cross_sell_total : 2D array
            array representing both cross_sell_target and cross_sell_self
        """
        """function that gives whether a customers outside the training set are eligible for a cross sell"""

        active_value = act_obj.active_value(self, param_act, n_segments_act, t, data)

        cross_sell_target, cross_sell_self, cross_sell_total = self.cross_sell_yes_no(param_cross, n_segments_cross, active_value, tresholds, order_active_high_to_low, data)

        return cross_sell_target, cross_sell_self, cross_sell_total



    def new_hessian(self,parameters, n_segments = None):
        if self.function_for_hessian == None:
            visualize_old = self.visualize_data
            self.visualize_data = False
            self.interpret_parameters(parameters,n_segments)
            self.visualize_data = visualize_old

        self.maximization_iters = 0
        change = 1e-10
        change2 = -1e-11
        n = parameters.shape[0]
        parameters_2 = parameters + change2

        value_0_1 = self.function_for_hessian(parameters, reg = self.reg_term)
        value_0_2 = self.function_for_hessian(parameters_2, reg = self.reg_term)

        grad_1 = np.full(n,np.nan)
        grad_2 = np.full(n,np.nan)
        for i in range(0,n):
            new_parameters = parameters
            new_parameters[i] = parameters[i] + change
            value_1_1 = self.function_for_hessian(new_parameters, reg = self.reg_term)
            grad_1[i] = (value_1_1 - value_0_1) / change

            new_parameters = parameters_2
            new_parameters[i] = parameters_2[i] + change2
            value_1_2 = self.function_for_hessian(new_parameters, reg = self.reg_term)
            grad_2[i] = (value_1_2 - value_0_2) / change2

        identity = np.identity(n)
        s = parameters - parameters_2
        y = grad_1 - grad_2
        rho = 1/ ( (y*s).sum() )

        hessian_inv =  (identity - rho * np.outer(s,y)).dot(identity - rho * np.outer(y,s)) + rho*np.outer(s,s)

        self.maximization_iters = 0
        return hessian_inv


    
    def get_standard_errors(self, param_in, n_segments):
        """Print the standard errors given certain input parameters"""
        
         #----------- Initialise everything so that we get the shapes-------------
        gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
        gamma_sr_0 =  np.ones( (n_segments-1,n_segments) )
        gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) ) 
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1))
   
        #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
        shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], 
                           [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)

        #----------------------- Do a single EM step --------------------------- 
        
        alpha_out, beta_out = self.forward_backward_procedure(param_in, shapes, n_segments)

        print("Doing BFGS M-step to get Hessian") 
        param_afterBFGS, hess_inv = self.maximization_step(alpha_out, beta_out, param_in, 

                                                 shapes, n_segments, self.reg_term,
                                                 self.max_method, bounded = None,
                                                 end = True)
        
        #----------------------- get sd's from hessian --------------------------- 
        diag = np.diag(hess_inv)
        se = np.sqrt(diag)
        
        # save the values to a dataframe and save to csv?
        df = pd.DataFrame(columns = ["source","parameter","se","t", "p-value"])
        df["parameter"] = param_afterBFGS
        df["se"] = se
        df["t"] = df["parameter"] / df["se"]
        
        degrees= (self.n_customers - len(param_in))
        for i in range(0,len(df)):
            tval = df.loc[i,"t"]
            df.loc[i,"p-value"] = 1-t.cdf(abs(tval),df = degrees)
            
        a = gamma_0.size
        b = gamma_sr_0.size
        c = gamma_sk_t.size
        d = beta.size
        df.loc[0:a,"source"] = "gamma_0"
        df.loc[a:a+b,"source"] = "gamma_sr_0"
        df.loc[a+b:a+b+c,"source"] = "gamma_sk_t"
        df.loc[a+b+c:a+b+c+d,"source"] = "beta"
        
        #----------------------- Save results --------------------------- 
        utils.save_df_to_csv(df, self.outdir, f"{self.outname}_standarderrors", 
                             add_time = False )
        
        logl_outafterBFGS = self.loglikelihood(param_afterBFGS, shapes, n_segments)
    
        with open(f'{self.outdir}/{self.outname}_afterBFGS.txt', 'w') as f:           
           f.write(f"time: {utils.get_time()} \n")
           f.write(f"dependent variable: {self.list_dep_var} \n")
           f.write(f"covariates: {self.list_covariates} \n")   
           f.write(f"number of covariates: {len(self.list_covariates)} \n\n")   
           arraystring = utils.printarray(param_afterBFGS)
           paramstring = f"param_out = np.array({arraystring}) \n\n"
           f.write(paramstring)
           f.write(f"LogLikelihood value: {logl_outafterBFGS}")

        # Also save the hessians in a file
        with open(f'{self.outdir}/{self.outname}_HESSIANBFGS_len{len(hess_inv)}.txt', 'w') as f:
            np.set_printoptions(threshold=np.inf) # so we can print the whole array?
            hesinvstring = utils.printarray(hess_inv, removenewlines = True)
            paramstring = f"Hessian_out = np.array({hesinvstring}) \n\n"
            f.write(paramstring)
    
        return hess_inv, df, param_afterBFGS
    
    
    
    def calculate_gini(self,prod_ownership_new, prod_ownership_old, product_probs, binary = True):
        """Calculate gini coefficient. If binary then prod ownership is 1 
        and 0 for each, if not then prod ownership should be ownership 
        numbers for each"""
    
        Gini = pd.DataFrame(columns = prod_ownership_new.columns) 
        if binary:
            for i in range(0, len(prod_ownership_new.columns)):
                prod_probs = product_probs[:,i,:] 
                
                # Get the households who did NOT have product in prev period
                n_i = len(prod_ownership_old[prod_ownership_old.iloc[:,i]==0]) 
                select = (prod_ownership_old.iloc[:,i]==0)
                col = prod_ownership_new.columns[i]
                 
                # Percentage of those households who now do own the product
                change = prod_ownership_new.loc[select,col] # todo check that this selects the right thing
                mu_i = (sum(change) / len(change))*100 # percentage that is 1
            
                # Get the sum of probabilities for >0 of the product
                prod_own = prod_probs[:,1:].sum(axis=1) 
                
                # Ranked probabilities - 
                # We want the person with the highest probability to get the lowest rank
                probranks = pd.DataFrame(prod_own).rank( ascending = False) #method = 'max'
                # NOW SELECT THE ONES THAT BELONG TO THE NON-OWNING GROUP
                probranks = probranks[select]
                
                sumrank = 0
                for k in range(0,len(probranks)): # we sum only over the select households?
                    #sumrank += probranks.iloc[k,0] * prod_ownership_new.loc[k,col]
                    sumrank += probranks.iloc[k,0] * change.reset_index(drop=True)[k]
                  
                Gini_i = 1 + (1/n_i) - ( 2 / ( (n_i**2)*mu_i  ) )*sumrank 
                Gini.loc[0,col] = Gini_i
                
        else: # the prod ownerships should be numbers of products
           for i in range(0, len(prod_ownership_new.columns)):
               # get the different possible values of ownerships
               values = pd.Series(prod_ownership_old.iloc[:,i].unique()).sort_values()
               prod_probs = product_probs[:,i,:] # get probs for this segment
               
               for j in values: 
                   # Number of households who did NOT have this exact number of products
                   n_i = len(prod_ownership_old[prod_ownership_old.iloc[:,i]!=j])
                   select = (prod_ownership_old.iloc[:,i]!=j)
                   col = prod_ownership_new.columns[i]
                   
                   # Make a dummy for # of products ownership in the new period
                   ownership_new_dummy = pd.Series(np.zeros(len(prod_ownership_new)))
                   ownership_new_dummy[prod_ownership_new.iloc[:,i] == j] = 1
                   ownership_new_dummy = ownership_new_dummy[select]
                  
                   # Percentage of the selected households who now do own the product
                   mu_i = (sum(ownership_new_dummy) / len(ownership_new_dummy))*100 # percentage that has changed
                   #TODO does this need to be *100 ????
                   
                   # Get the sum of probabilities for exactly j of the product
                   prod_own = prod_probs[:,int(j)]
    
                   # Ranked probabilities - 
                   # We want the person with the highest probability to get the lowest rank
                   probranks =pd.DataFrame(prod_own).rank(ascending = False) #method='max', 
                   # NOW SELECT THE ONES THAT BELONG TO THE NON-OWNING GROUP
                   probranks = probranks[select]
                
                   sumrank = 0
                   for k in range(0,len(probranks)):
                       sumrank += probranks.iloc[k,0] * ownership_new_dummy.iloc[k]
                  
                   Gini_i = 1 + (1/n_i) - ( 2 / ( (n_i**2)*mu_i  ) )*sumrank 
                
                   Gini.loc[int(j),col] = Gini_i               
        return Gini
    
    
    def calculate_accuracy(self, cross_sell_pred,cross_sell_true, print_out = True):
        """Calculate a few ealuation measures including accuracy"""    
        FPvec = []
        TPvec = []
        FNvec = []
        TNvec = []
        accuracyvec = []
        sensitivityvec = []
        
        for i in range(0, len(cross_sell_true.columns)):
          pred_labels = cross_sell_pred[:,i]
          true_labels = cross_sell_true.iloc[:,i] # check that this selects the right things
          
          # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
          TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
          # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
          TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
          # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
          FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
          # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
          FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
          
          TPvec.append(TP)
          TNvec.append(TN)
          FPvec.append(FP)
          FNvec.append(FN)    
          accuracy = (TP+TN) / (TP+TN+FP+FN)
          sensitivity = TP / (TP+FN)
          #specificity = TN/ (TN+FP)
          #precision = TP / (TP+FP) # Note: goes wrong if nothing is predicted positive
          accuracyvec.append(accuracy)
          sensitivityvec.append(sensitivity)
         
        if print_out:
            print(f"Accuracy output: {accuracyvec}")
            print(f"Sensitivity output: {sensitivityvec}")
            print(f"TP: {TPvec}")
            print(f"TN: {TNvec}")
            print(f"FP: {FPvec}")
            print(f"FN: {FNvec}")
            
        # PUT THE RESULTS INTO A DATAFRAME
        evaluation = pd.concat([pd.Series(accuracyvec), pd.Series(sensitivityvec),
                                pd.Series(TPvec), 
                                pd.Series(TNvec), pd.Series(FPvec), 
                                pd.Series(FNvec)], axis = 1).transpose()

        evaluation.columns = cross_sell_true.columns
        evaluation["measure"] = ["accuracy","sensitivity","TP","TN","FP","FN"]
        
        return evaluation
         
  
    
    def hypo_customers(self, activity, param, n_segments, interdir):
        
         #----------- Initialise everything so that we get the shapes-------------
        gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
        gamma_sr_0 =  np.ones( (n_segments-1,n_segments) )
        gamma_sk_t =  np.ones( (n_segments-1,self.n_covariates) ) 
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1))
   
        #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
        shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], 
                           [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)        
           
        if activity == False:
            n_possible = 5*5*2*5*7*6*2
            
            inc_dummy = np.zeros((n_possible, 4))
            age_dummy = np.zeros((n_possible, 4))
            gender_dummy = np.zeros((n_possible, 1))
            hh_dummy = np.zeros((n_possible, 4))
            saldo_dummy = np.zeros((n_possible, 6))
            bus_dummy = np.zeros((n_possible, 5))
            rest = np.zeros((n_possible, 2))
            
            
            div = int(n_possible/2)
            gender_dummy[div : 2*div, 0] = np.ones(div)
            
            for i in range(1,5):
                div = int(n_possible/5)
                inc_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
                age_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
                hh_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
    
            for i in range(1,6):
                div = int(n_possible/6)
                bus_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
              
            for i in range(1,7):
                div = int(n_possible/7)
                saldo_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
                
            Z = np.hstack((rest, inc_dummy,age_dummy,gender_dummy,hh_dummy, saldo_dummy,bus_dummy))
                                    
            Z = pd.read_csv(f"{interdir}/hypo_data.csv")
            Z =  Z.to_numpy()          

                 # Income     Age     Gen HHsize     Saldo        Business
            Z = np.array([[0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],  #Segment 1
              [0,0, 0,1,0,0,  1,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,1,0,0,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],              
              [0,0, 0,1,0,0,  0,0,1,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,1,0,  1,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,1,0,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,1,0,  0,0,1,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,1,  1,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,1,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,1,  0,0,1,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  1,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,1,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 1,0,0,0,  1,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 1,0,0,0,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 1,0,0,0,  0,0,1,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0], #segment 2
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,1,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 1,0,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,1,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,1],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 1,0,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 0,1,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 0,0,0,0,1],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,1,0,0,  0,0,0,0,0,0, 1,0,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,1,0,0,  0,0,0,0,0,0, 0,1,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,1,0,0,  0,0,0,0,0,0, 0,0,1,0,0],
              [0,0, 0,0,0,0,  0,1,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],

              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,0,0,0,0], #segment 3          
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,0,0,0,0],
              
              [0,0, 0,1,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0], #segment 4
              [0,0, 0,0,0,1,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,0,0,0,0],
              [0,0, 0,1,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,1,0,0,  0,0,0,0,  0,  1,0,0,0,  0,1,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,1,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,1,0,0,0, 0,0,0,0,0],              
              [0,0, 0,1,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,0,0,0,0],
              [0,0, 0,1,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,1,0,0,0,0, 0,0,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  1,0,0,0,  0,0,1,0,0,0, 0,0,0,0,0],
    
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 1,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,1,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,1,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,1,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 1,0,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,1,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,0,1,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,1,0,0,0,0, 0,0,0,1,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 1,0,0,0,0],              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,1,0,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,0,1,0,0],
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,1,0,0,0, 0,0,0,1,0],   
              
              [0,0, 0,0,0,0,  0,0,0,0,  0,  0,0,0,0,  0,0,0,0,0,0, 0,0,0,0,0]])
            
            P_s_given_Z = ef.prob_P_s_given_Z(self, param, shapes, Z, n_segments)
            P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments)

            P_s_given_r = np.swapaxes(P_s_given_r, 0, 2)
            P_s_given_r = np.swapaxes(P_s_given_r, 1, 2)
            return P_s_given_Z, P_s_given_r, Z

        else:
            n_possible = 5*5*2
            
            inc_dummy = np.zeros((n_possible, 4))
            age_dummy = np.zeros((n_possible, 4))
            gender_dummy = np.zeros((n_possible, 1))
            
            div = int(n_possible/2)
            gender_dummy[div : 2*div, 0] = np.ones(div)
            
            for i in range(1,5):
                div = int(n_possible/5)
                inc_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
                age_dummy[i*div : (i+1)*div, i-1] = np.ones(div)
    

            Z = np.hstack((inc_dummy,age_dummy,gender_dummy))
            
            #alpha, beta = self.forward_backward_procedure(param_in, shapes, n_segments)
            #P_s_given_Y_Z = ef.state_event(self, alpha, beta)

            #return P_s_given_Y_Z



    # def new_hessian(self,parameters, n_segments = None):
    #     if self.function_for_hessian == None:
    #         visualize_old = self.visualize_data
    #         self.visualize_data = False
    #         self.interpret_parameters(parameters,n_segments)
    #         self.visualize_data = visualize_old
    #
    #     self.maximization_iters = 0
    #     change = 1e-10
    #     n = parameters.shape[0]
    #     value_0 = self.function_for_hessian(parameters, reg = self.reg_term)
    #     hessian = np.full((n, n), np.nan)
    #     for i, outer_number in enumerate(parameters):
    #         outer_parameters = parameters
    #         outer_parameters[i] = outer_number + change
    #         value_1_outer = self.function_for_hessian(outer_parameters, reg = self.reg_term)
    #         gradient_val_outer = (value_1_outer - value_0) / change
    #         for j, inner_number in enumerate(outer_parameters):
    #             inner_parameters = outer_parameters
    #             inner_parameters[j] = inner_number + change
    #             value_1_inner = self.function_for_hessian(inner_parameters, reg = self.reg_term)
    #             gradient_val_inner = (value_1_inner - value_1_outer) / change
    #             hessian[j, i] = gradient_val_inner - gradient_val_outer
    #     self.maximization_iters = 0
    #
    #     hessian_inv = np.linalg.inv(hessian)
    #     return hessian_inv
        