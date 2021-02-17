# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:32:43 2021

@author: matth
"""

# -*- coding: utf-8 -*-
"""
This code aims to execute a Baum-Welch/forward-backward algorithm to estimate a Hidden Markov Model 
(with or without modelling the transition and initialisation probabilities by covariates) 

@author: Matthijs van Heese
"""
import extra_functions_HMM_eff as ef
import numpy as np 
from scipy.optimize import minimize 
import utils
from tqdm import tqdm
from time import perf_counter
#from geneticalgorithm import geneticalgorithm as ga

#from pyswarm import pso

class HMM_eff:
    
    def __init__(self, list_dataframes, list_dep_var, 
                   list_covariates = [], covariates = False):
        """Initialisation of a HMM object
           list_dataframes: list consisting of the timeperiod-specific dataframes
           list_dep_var: list consisting of all the names of the variables we use as dependent variables
           list_covariates: list consisting of all the names of the variables we use as covariates
           covariates: boolean that indicates whether transition/state probabilities are modelled as logit model"""
           
        self.list_dataframes = list_dataframes
        self.list_dep_var = list_dep_var
        self.list_covariates = list_covariates
        
        self.n_covariates = len(list_covariates) #initialise the number of covariates
        self.n_customers = self.list_dataframes[0].shape[0] #initialise the number of customers
        self.n_products = len(list_dep_var) #initialise the number of product
        self.T = len(list_dataframes) #initialise the number of dataframes, thus the timeperiod
        
        self.covariates = covariates #initialise whether covariates are used to model the transition/state probabilities

        #compute per dependent variable the number of categories (possible values)
        self.n_categories = np.zeros((self.n_products))
        for i in range(0,self.T):
            for j in range(0,self.n_products):
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
          
    def EM(self, n_segments, tolerance = 10**(-4), max_method = "BFGS"):
        """function to run the EM algorithm
            n_segments: number of segments to use for the estimation of the HMM
            tolerance: convergence tolerance
            max_method: maximization method to use for the maximization step"""
        
        if self.covariates == True:         #initialise parameters for HMM with the probabilities as logit model
            """gamma_0 = 0.2 * np.ones( (n_segments-1, self.n_covariates+1) ) #parameters for P(S_0 = s|Z)
            gamma_sr_0 = 0.3 * np.ones( (n_segments-1,n_segments) ) #parameters for P(S_t = s | S_t-1 = r)
            gamma_sk_t = 0.4 * np.ones( (n_segments-1,self.n_covariates) )  #parameters for P(S_t = s | S_t-1 = r)
            beta = 0.5 * np.ones((n_segments, self.n_products, max(self.n_categories))) #parameters for P(Y| S_t = s)
            #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
            shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
            param = ef.param_matrices_to_list(self, gamma_0 = gamma_0, gamma_sr_0 = gamma_sr_0, gamma_sk_t = gamma_sk_t, beta = beta)  #convert parametermatrices to list
            param_out = param #set name of parameterlist for the input of the algorithm """
            gamma_0 = 0.2 * np.ones( (n_segments-1, self.n_covariates+1) ) #parameters for P(S_0 = s|Z)
            gamma_sr_0 = 0.3 * np.ones( (n_segments-1,n_segments) ) #parameters for P(S_t = s | S_t-1 = r)
            gamma_sk_t = 0.4 * np.ones( (n_segments-1,self.n_covariates) )  #parameters for P(S_t = s | S_t-1 = r)
            beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)
            
            for s in range(n_segments):
                for p in range(0,self.n_products):
                    beta[s,p,0:self.n_categories[p]-1] = 0.5 * np.ones((1,self.n_categories[p]-1))                    
            
            #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
            shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
            param = ef.param_matrices_to_list(self, n_segments, gamma_0 = gamma_0, gamma_sr_0 = gamma_sr_0, gamma_sk_t = gamma_sk_t, beta = beta)  #convert parametermatrices to list
            param_out = param #set name of parameterlist for the input of the algorithm
            
        else:         #initialise parameters for HMM without the probabilities as logit model
            A = 1/n_segments * np.ones((n_segments-1,n_segments)) #parameters of P(S_t = s | S_t-1 = r)
            pi = 1/n_segments * np.ones((n_segments-1))  #parameters for P(S_0 = s)
            b = np.ones((n_segments, self.n_products, max(self.n_categories))) ##parameters for P(Y| S_t = s)
            #shapes indicate the shapes of the parametermatrices, such that parameters easily can be converted to 1D array and vice versa
            shapes = np.array([[A.shape,A.size], [pi.shape, pi.size], [b.shape, b.size]], dtype = object)
            param = ef.param_matrices_to_list(self, n_segments, A = A, pi = pi, b = b) #convert parametermatrices to list
            param_out = param #set name of parameterlist for the input of the algorithm

        #initialise
        iteration = 0
        difference = True
        
        print(f"****Starting EM prodecure, at {utils.get_time()}****")
        print(f"tolerance: {tolerance}")
        print(f"number of parameters: {len(param_out)}")
        
        start_EM = utils.get_time()
        
        #Start EM procedure
        while difference:
                
            param_in = param_out #update parameters
            
            start1 = utils.get_time() 
            
            #perform forward-backward procedure (expectation step of EM) 
            alpha, beta = self.forward_backward_procedure(param_in, shapes, n_segments)
              
            start = utils.get_time() #set start time to time maximisation step
            print(f"E-step duration: {utils.get_time_diff(start,start1)} ")

            #perform maximisation step 
            opt_result = self.maximization_step(alpha, beta, param_in, shapes, n_segments, max_method)
            param_out = opt_result.x
            print(param_out)
            
            end = utils.get_time()#set start time to time maximisation step
            diff = utils.get_time_diff(start,end)#get difference of start and end time, thus time to run maximisation 
            print(f"Finished iteration {iteration}, duration M step {diff}")

            difference = (np.max(abs(param_in-param_out)) > tolerance) #set difference of input and output of model-parameters
            print(f"max difference: {np.max(abs(param_in-param_out))}")
            if iteration == 0:
                print('hoi')
                
            iteration = iteration + 1 #update iteration
        
        end_EM = utils.get_time()
        diffEM = utils.get_time_diff(start_EM,end_EM)
        print(f"Total EM duration: {diffEM}")
        
        if self.covariates == True:
            return param_out, alpha, shapes
        else:
            return param_out, shapes

        
        
    #------------Function for the expectation step------------
        
    def forward_backward_procedure(self, param, shapes, n_segments):
        """function for the expectation step: compute alpha and beta with all parameters"""
    
        p_js = ef.prob_p_js(self, param, shapes, n_segments)
            
        alpha = np.zeros((n_segments, self.n_customers, self.T))
        beta = np.zeros((n_segments, self.n_customers, self.T))
            
        
        for t in range(0,self.T):
            v = self.T - t - 1
                        
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]
            else:
                Z = []
                        
            P_y_given_s = ef.prob_P_y_given_s(self, Y, p_js, n_segments) #i x s 
                            
            if t == 0: #s x i x t
                P_s_given_Z = ef.prob_P_s_given_Z(self, param, shapes, Z, n_segments) #[i x s]
                alpha[:,:,t] = np.transpose( np.multiply(P_y_given_s, P_s_given_Z) ) # [i x s] [i x s] = [i x s]'
                beta[:,:,v] = np.ones((n_segments,self.n_customers))
            else:
                P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments) #[i x s x s]
                
                sum_alpha = np.zeros( (n_segments, self.n_customers) )
                sum_beta = np.zeros( (n_segments, self.n_customers) )
                for r in range(0,n_segments):                               #[ix1]                                               [ixs]               [ixs]
                    sum_alpha = sum_alpha + np.transpose( np.multiply( np.transpose(np.array([alpha[r,:,t-1]])) , np.multiply(P_s_given_r[:,:,r], P_y_given_s) ) )
                    sum_beta = sum_beta + np.transpose( np.multiply( np.transpose(np.array([beta[r,:,v+1]])) , np.multiply(P_s_given_r[:,r,:] , np.transpose(np.array([P_y_given_s[:,r]]))) ) )
                                                                           #[i,1]                                            [i,s]                   [i,1]                            
                alpha[:,:,t] = sum_alpha
                beta[:,:,v]  = sum_beta
                        
        return alpha, beta


    
      
    def maximization_step(self, alpha, beta, param_in, shapes, n_segments, max_method):
        """
        

        Parameters
        ----------
        alpha : TYPE
            DESCRIPTION.
        beta : TYPE
            DESCRIPTION.
        param_in : TYPE
            DESCRIPTION.
        shapes : TYPE
            DESCRIPTION.
        n_segments : TYPE
            DESCRIPTION.
        max_method : TYPE
            DESCRIPTION.

        Returns
        -------
        param_out : TYPE
            DESCRIPTION.

        """
        """function for the maximization step"""
            
        P_s_given_Y_Z = ef.state_event(self, alpha, beta, n_segments)

        x0 = param_in
            
        """perform the maximization"""

        self.maximization_iters = 0
        self.iterprint = False

        minimize_options = {'disp': True, 'fatol': 1e-2, 'xatol': 1}
        t1 = perf_counter()
        param_out = minimize(self.optimization_function, x0, args=(alpha, beta, param_in, shapes, n_segments, P_s_given_Y_Z), method=max_method,
                             options= minimize_options)
        t2 = perf_counter()
        print('Time for maximization:',t2-t1, 'with ', self.maximization_iters, "number of iterations")
        pass

        return param_out
        #param_out = pso(self.optimization_function, args=(alpha, beta, param_in, shapes, n_segments))
    
    def optimization_function(self, x, alpha, beta, param_in, shapes, n_segments, P_s_given_Y_Z):
        """function that has to be minimized"""
                    
        
        p_js_max = ef.prob_p_js(self, x, shapes, n_segments)
        p_js_cons = ef.prob_p_js(self, param_in, shapes, n_segments)
        P_s_given_Y_Z_ut = np.multiply(alpha, beta)

        """compute function"""
        sum = 0;

        Y = self.list_Y[0]
        if self.covariates == True:
            Z = self.list_Z[0]
        else: 
            Z = np.array([])
            
        P_s_given_Y_Z_0 = np.transpose(P_s_given_Y_Z[:,:,0]) #s x i x t
        
        #t=0, term 1
        P_s_given_Z = ef.prob_P_s_given_Z(self, x, shapes, Z, n_segments)  #i x s
        sum = sum + np.sum(np.multiply(P_s_given_Y_Z_0, np.log(P_s_given_Z + 10**(-300))))
        
        #t=0, term 3
        P_y_given_s = ef.prob_P_y_given_s(self, Y, p_js_max, n_segments) #ixs
        sum = sum + np.sum(np.multiply(P_s_given_Y_Z_0, np.log(P_y_given_s + 10**(-300))))
        
        
        for t in range(1,self.T):
            Y = self.list_Y[t]
            if self.covariates == True:
                Z = self.list_Z[t]   
            else: 
                Z = []
    
            #t=t, term 2
            #for r in range(0,n_segments):
            # These do not depend on the segment - use as input for joint event function
            P_s_given_r_cons = ef.prob_P_s_given_r(self, param_in, shapes, Z, n_segments)
            P_y_given_s_cons = ef.prob_P_y_given_s(self, Y, p_js_cons, n_segments)
            
            P_s_given_r_max = ef.prob_P_s_given_r(self, x, shapes, Z, n_segments)
            
            for s in range(0,n_segments):
                P_sr_given_Y_Z = ef.joint_event(self, Y, Z, alpha, beta, param_in, shapes, t, s, n_segments,
                                                P_s_given_Y_Z_ut, P_s_given_r_cons, P_y_given_s_cons)
                #sum = sum + np.sum( np.multiply(P_sr_given_Y_Z, np.log(P_s_given_r[:,s,r]))  )
                sum = sum + np.sum( np.multiply(P_sr_given_Y_Z, np.log(P_s_given_r_max[:,s,:] + 10**(-300)))  )

            #t=t, term 3
            P_y_given_s_max = ef.prob_P_y_given_s(self, Y, p_js_max, n_segments) #ixs
            P_s_given_Y_Z_t = np.transpose(P_s_given_Y_Z[:,:,t]) #ixs
            sum = sum + np.sum(np.multiply(P_s_given_Y_Z_t, np.log(P_y_given_s_max + 10**(-300))))

        self.maximization_iters += 1
        if self.iterprint:
            print('function value:', -sum,' at iteration ',self.maximization_iters)
        return -sum

    def predict_product_ownership(self, param, shapes, n_segments, alpha):
        if self.covariates == True:
            Z = self.list_Z[self.T-1]

            gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(self, n_segments, param, shapes)

            p_js = ef.prob_p_js(self, param, shapes, n_segments) #s x p x c
            P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z)
            nextstate = np.zeros(self.n_customers, n_segments) #i x s
            prediction = np.zeros(self.n_customers, self.products, max(self.n_categories))


            for i in range(0,self.n_customers):

                probstate = alpha[i,self.T,:] / np.sum(alpha[:, i, self.T])

                for s in  range(1,n_segments):
                    nextstate[i,:] += probstate * P_s_given_r[i,:,s]

                    #i x s    s x p x c
            for p in range(1,self.n_products):
                for c in range(1,self.n_categories[p]):
                    for s in range(1,n_segments):
                        prediction[i,p,c] += nextstate[i,s]* p_js[s,p,c]

            #prediction = np.einsum('is,spc->ipc', nextstate, p_js)

            return prediction

    def active_value(self, param, shapes, n_segments):
        if self.covariates == False:
            Y = self.list_Y[self.T-1]

            p_js = ef.prob_p_js(self, param, shapes, n_segments)
            P_Y_given_S = ef.prob_P_y_given_s(self, Y, p_js, n_segments)

            active_value = np.argmax(P_Y_given_S, axis=1)

            return active_value


    def cross_sell_yes_no(self, param, shapes, n_segments, alpha, active_value, tresholds):


        prod_own = self.predict_product_ownership(param, shapes, n_segments, alpha)
        Y = self.list_Y[self.T-1]

        expected_n_prod = np.zeros(self.n_customers, self.n_products)

        dif_exp_own = np.zeros(self.n_customers, self.n_products)
        
        cross_sell_target = np.zeros(self.n_customers, self.n_products)
        cross_sell_self = np.zeros(self.n_customers, self.n_products)
        cross_sell_total = np.zeros(self.n_customers, self.n_products)
        

        for i in range(0, self.n_customers):
            for p in range(0,self.n_products):
                for c in range(0,self.n_categories[p]):
                    expected_n_prod[i,p] = expected_n_prod[i,p] + c*prod_own[i,p,c]
                    
                dif_exp_own[i,p] = expected_n_prod[i,p] - Y[i,p]
            if dif_exp_own[i,p] >= tresholds[0]:
                if active_value == 2:
                    cross_sell_target[i,p] = False
                    cross_sell_self[i,p] = True
                    cross_sell_total = True
                if active_value == 1:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True               
                if active_value == 0:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True  
            elif dif_exp_own[i,p] < tresholds[0] & dif_exp_own >= tresholds[1]:
                if active_value == 2:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True
                if active_value == 1:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True               
                if active_value == 0:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True 
            else:
                if active_value == 2:
                    cross_sell_target[i,p] = True
                    cross_sell_self[i,p] = False
                    cross_sell_total = True
                if active_value == 1:
                    cross_sell_target[i,p] = False
                    cross_sell_self[i,p] = False
                    cross_sell_total = False               
                if active_value == 0:
                    cross_sell_target[i,p] = False
                    cross_sell_self[i,p] = False
                    cross_sell_total = False 
                        
        return cross_sell_target, cross_sell_self, cross_sell_total
                   
        
        
              
            
            
            
            
