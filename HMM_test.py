# -*- coding: utf-8 -*-
"""
This code aims to execute a Baum-Welch/forward-backward algorithm to estimate a Hidden Markov Model 
(with or without modelling the transition and initialisation probabilities by covariates) 

@author: Matthijs van Heese
"""
import extra_functions_HMM as ef
import numpy as np 
from scipy.optimize import minimize 
import math
import eff_HMM as eh
import utils
from pyswarm import pso

class HMM:
    
    def __init__(self, list_dataframes, list_dep_var, 
                   list_covariates = [], covariates = False):
        """list_dataframes: list consisting of the timeperiod-specific dataframes
           list_dep_var: list consisting of all the names of the variables we use as dependent variables
           list_covariates: list consisting of all the names of the variables we use as covariates
           covariates: boolean that indicates whether transition/state probabilities are modelled as logit"""
           
        self.list_dataframes = list_dataframes
        self.list_dep_var = list_dep_var
        self.list_covariates = list_covariates
        
        self.n_dep_var = len(list_dep_var)
        self.n_covariates = len(list_covariates)
        self.n_customers = self.list_dataframes[0].shape[0]
        self.n_products = len(list_dep_var)
        self.T = len(list_dataframes)
        
        self.covariates = covariates
        
        """compute per dependent variable the number of categories"""
        self.n_categories = np.zeros((self.n_dep_var))
        for i in range(0,self.T):
            for j in range(0,self.n_dep_var):
                n_per_df = self.list_dataframes[i][list_dep_var[j]].nunique();
                if n_per_df > self.n_categories[j]:
                    self.n_categories[j] = n_per_df
        self.n_categories = self.n_categories.astype(int)
                    
        """Compute lists consisting of Y and Z matrices per timeperiod"""
        self.list_Y =[]

        if covariates == True:
            self.list_Z = []

        for i in range(0,self.T):
            if covariates == True:
                Z = list_dataframes[i][list_covariates]
                self.list_Z.append(Z.to_numpy())
            Y = list_dataframes[i][list_dep_var]
            self.list_Y.append(Y.to_numpy())
        
        
          
    def EM(self, n_segments, tolerance = 10**(-4), max_method = "BFGS"):
        """function to run the EM algorithm
            n_segments: number of segments to use for the estimation of the HMM
            tolerance: convergence tolerance
            max_method: maximization method to use for the maximization step"""


        if self.covariates == True:
            gamma_0 = 0.2 * np.ones( (n_segments-1, self.n_covariates+1) ) #parameters of P(S_0 = s|Z)
            gamma_sr_0 = 0.3 * np.ones( (n_segments-1,n_segments) ) #parameters of P(S_t = s | S_t+1 = r)
            gamma_sk_t = 0.4 * np.ones( (n_segments-1,self.n_covariates) )  #parameters of P(S_t = s | S_t+1 = r)
            beta = 0.5 * np.ones((n_segments, self.n_products, max(self.n_categories))) #parameters of P(Y|s)
            shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], [gamma_sk_t.shape, gamma_sk_t.size], [beta.shape, beta.size]], dtype = object)
            param = ef.param_matrices_to_list(self, gamma_0 = gamma_0, gamma_sr_0 = gamma_sr_0, gamma_sk_t = gamma_sk_t, beta = beta)  
            param_out = param
        else: 
            A = 1/n_segments * np.ones((n_segments,n_segments)) #P(Y_it | X_it = s)
            pi = 1/n_segments * np.ones((n_segments))  #P(X_i0 = s| Z_i0)
            b = np.ones((n_segments, self.n_products, max(self.n_categories))) #parameters for P(Y_it | X_it = s)
            shapes = np.array([[A.shape,A.size], [pi.shape, pi.size], [b.shape, b.size]], dtype = object)
            param = self.param_matrices_to_list(A = A, pi = pi, b = b)
            param_out = param

        iteration = 0
        difference = True
        
        """Start EM procedure"""
        while difference:
                
            param_in = param_out
                
            alpha, beta = self.forward_backward_procedure(param_in, shapes, n_segments)
              
            start = utils.get_time()

            param_out = eh.maximization_step(self, alpha, beta, param_in, shapes, n_segments, max_method)
            
            
            end = utils.get_time()
            diff = utils.get_time_diff(start,end)
            iteration = iteration + 1
                            
            difference = (any(abs(param_in-param_out)) > tolerance) 
        
        return self.param_list_to_matrices(param, shapes)
        
        
    #------------Function for the expectation step------------
        
    def forward_backward_procedure(self, param, shapes, n_segments):
        """function for the expectation step: compute alpha and beta with all parameters"""

        p_js = ef.prob_p_js(self, param, shapes, n_segments)
        
        alpha = np.zeros((n_segments, self.n_customers, self.T))
        beta = np.zeros((n_segments, self.n_customers, self.T))
    
        for i in range(0,self.n_customers):
            for t in range(0,self.T):
                v = self.T - t - 1
                
                Y = np.array([self.list_Y[t][i,:]])
                if self.covariates == True:
                    Z = np.array([self.list_Z[t][i,:]])
                else:
                    Z = []
                
                P_y_given_s = ef.prob_P_y_given_s(self, Y, p_js, n_segments)
                
                if t == 0:
                    P_s_given_Z = ef.prob_P_s_given_Z(self, param, shapes, Z, n_segments)
                    alpha[:,i,t] = np.multiply(P_y_given_s, P_s_given_Z).flatten()
                    beta[:,i,v] = np.ones((n_segments))
                else:
                    P_s_given_r = ef.prob_P_s_given_r(self, param, shapes, Z, n_segments)
    
                    sum_alpha = np.zeros( (n_segments) )
                    sum_beta = np.zeros( (n_segments) )
                    for r in range(0,n_segments):
                        sum_alpha = sum_alpha + np.multiply( np.multiply(alpha[:,i,t-1],P_s_given_r[:,r]), P_y_given_s.flatten())
                        sum_beta = sum_beta + np.multiply( np.multiply(beta[:,i,v+1],P_s_given_r[r,:]), P_y_given_s.flatten())
                    alpha[:,i,t] = sum_alpha
                    beta[:,i,v]  = sum_beta
                
        return alpha, beta
    
      
    def maximization_step(self, alpha, beta, param_in, shapes, n_segments, max_method):
        """function for the maximization step"""
        
        #x0 = param_in
        
        """perform the maximization"""
        #param_out = minimize(self.optimization_function, x0, args=(alpha, beta, param_in, shapes, n_segments), method=max_method)
        
        param_out = pso(self.optimization_function, args=(alpha, beta, param_in, shapes, n_segments))  
        
        return param_out
    
    
    def optimization_function(self, x, alpha, beta, param_in, shapes, n_segments):
        """function that has to be minimized"""
                
        
        """compute function"""
        sum = 0;
        
        for i in range(0,self.n_customers):   
            
            Y = self.list_Y[0][i,:]
            if self.covariates == True:
                Z = self.list_Z[0][i,:]    
            else: 
                Z = np.array([])
            
            P_s_given_Z = ef.prob_P_s_given_Z(self, x, shapes, Z, n_segments) 
            P_s_given_Y_Z = ef.state_event(self, alpha, beta, i, 0, n_segments)
            
            sum = sum + np.sum(np.multiply(P_s_given_Y_Z, np.log(P_s_given_Z)))
            
            for t in range(1,self.T):
                Y = self.list_Y[t][i,:]
                if self.covariates == True:
                    Z = self.list_Z[t][i,:]    
                else: 
                    Z = []
             
                P_s_given_r = ef.prob_P_s_given_r(self, x, shapes, Z, n_segments)
    
                for r in range(0,n_segments):
                    for s in range(0,n_segments):
                        P_sr_given_Y_Z = ef.joint_event(self, Y, Z, alpha, beta, param_in, shapes, i, t, s, r, n_segments)
            
                        sum = sum + P_sr_given_Y_Z * math.log(P_s_given_r[s,r])
    
            for t in range(0,self.T):
                P_y_given_s = ef.prob_P_y_given_s(self, Y, ef.prob_p_js(self, x, shapes, n_segments), n_segments)
                for s in range(0,n_segments):
                    P_s_given_Y_Z = ef.state_event(self, alpha, beta, i, t, n_segments)
                    
                    sum = sum + P_s_given_Y_Z[s] * math.log(P_y_given_s[s])
    
        return -sum    

    

    def predict(self, alpha, Z, gamma_sr_0, gamma_sk_t, pi, n_segments):
        prediction = np.zeros(self.n_customers, self.n_categories)
        T = self.T
        nextstate = np.zeros(self.n_customers, self.n_categories)
        for i in range(0,self.n_customers):
            probstate = []
            for s in range(0,n_segments):
                probstate[s] = alpha[i,T,s] / sum(alpha[i,T,1:n_segments]) 
            for a in  range(1,n_segments):
                nextstate[i,s] += probstate[a]* self.prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z)
            for j in range(1,self.n_categories):
                prediction[i,j] += nextstate[i,s]*self.prob_P_y_given_s(j, pi)
                
        return prediction

            
            
            
            
            
            
            
            
            
            
