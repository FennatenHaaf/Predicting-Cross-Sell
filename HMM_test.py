# -*- coding: utf-8 -*-
"""
This code aims to execute a Baum-Welch/forward-backward algorithm to estimate a Hidden Markov Model 
(with or without modelling the transition and initialisation probabilities by covariates) 

@author: Matthijs van Heese
"""

import numpy as np 
import scipy.stats as sci
import math

class HMM:
    
    
    def __init__(self, list_dataframes, list_dep_var, 
                   list_covariates = [], covariates = False):
        """list_dataframes: list consisting of the timeperiod-specific dataframes
           list_dep_var: list consisting of all the names of the variables we use as dependent variables
           list_covariates: list consisting of all the names of the variables we use as covariates
           covariates: boolean that indicates whether transition/state probabilities are modelled as logit"""
           
        self.list_dataframes = list_dataframes
        self.list_dep_var = list_dep_var
        self.n_dep_var = len(list_dep_var)
        self.list_covariates = list_covariates
        self.n_covariates = len(list_covariates)
        self.T = len(list_dataframes)
        self.n_customers = self.list_dataframes[0].shape[0]
        self.covariates = covariates
        
        """compute per dependent variable the number of categories"""
        self.n_categories = np.zeros((self.n_dep_var))
        for i in range(0,self.T):
            for j in range(0,self.n_dep_var):
                n_per_df = self.list_dataframes[i][list_dep_var[j]].nunique();
                if n_per_df > self.n_categories[j]:
                    self.n_categories[j] = n_per_df
                    
                    
        """Compute lists consisting of Y and Z matrices per timeperiod"""
        self.list_Y = np.array((self.T))

        if covariates:
            self.list_Z = np.array((self.T))       

        for i in range(0,self.T):
            if covariates:
                Z = list.dataframes[i][list_covariates]
                self.list_Z[i] = Z.to_numpy()
            Y = list.dataframes[i][list_dep_var]
            self.list_Y[i] = Y.to_numpy()
        
        
          
    def EM(self, n_segments, tolerance = 10**(-4), max_method = "BFGS"):
        """function to run the EM algorithm
            n_segments: number of segments to use for the estimation of the HMM
            tolerance: convergence tolerance
            max_method: maximization method to use for the maximization step"""


        """HMM with covariates"""
        if self.covariates = True:
        
            """Construct parameters to estimate"""
            #parameters of P(S_0 = s|Z)
            gamma_0 = np.ones( (n_segments-1, self.n_covariates+1) )
                        
            #parameters of P(S_t = s | S_t+1 = r)
            gamma_sr_0 = np.ones( (n_segments-1,n_segments) )
            gamma_sk_t = np.ones( (n_segments-1,self.n_covariates) )
                        
            #parameters of P(Y|s)
            beta = np.ones((n_segments, self.n_products, max(self.n_categories)))
        
            """Initialise parameters to estimate"""
            gamma_0_out = gamma_0
            gamma_sr_0_out = gamma_sr_0
            gamma_sk_t_out = gamma_sk_t
            beta_out = beta
            
            shapes = np.array([[gamma_0.shape,gamma_0.size], [gamma_sr_0.shape, gamma_sr_0.size], [gamma_sk_t_out.shape, gamma_sk_t_out.size], [beta_out.shape, beta_out.size]])
            param = param_matrices_to_list(self, gamma_0 = gamma_0, gamma_sr_0 = gamma_sr_0, gamma_sk_t = gamma_sk_t, beta = beta):

            
        """HMM without covariates"""
        else:
            
            """Construct parameters to estimate"""
            A = 1/n_segments * np.ones((n_segments,n_segments)) #P(Y_it | X_it = s)
            pi = 1/n_segments * np.ones((n_segments))  #P(X_i0 = s| Z_i0)
            b = np.ones((n_segments, self.n_products, max(self.n_categories))) #parameters for P(Y_it | X_it = s)
            
            """Initialise parameters to estimate"""
            A_out = A
            pi_out = pi
            b_out = b
            
            shapes = np.array([[A.shape,A.size], [pi.shape, pi.size], [b.shape, b.size]])
            param = param_matrices_to_list(self, A = A, pi = pi, b = b):


        iteration = 0
        difference = True
        
        """Start EM procedure"""
        while difference:
                
            param_in = param_out
                
            alpha,beta = self.forward_back_procedure(param_in, shapes, n_segments)
                
            param_out = self.maximization_step(alpha, beta, param_in, shapes, n_segments)
                
            iteration = iteration + 1
                            
            difference = (any(abs(param_in-param_out)) > tolerance) 
        
        return param_list_to_matrices(self, param, shapes):
        
        
    #------------Function for the expectation step------------
        
    def forward_backward_procedure(self, param, shapes, n_segments):
        """function for the expectation step: compute alpha and beta with all parameters"""

        p = self.prob_p_js(param, shapes)
        
        alpha = np.zeros(n_segments, self.n_customers, self.T)
        beta = np.zeros(n_segments, self.n_customers, self.T)
    
        for i in range(0,self.n_customers):
            for t in range(0,self.T):
                v = self.T - t
                
                Y = self.list_Y[t][i,:]
                if self.covariates = True:
                    Z = self.list_Z[t][i,:]
                else:
                    Z = np.array([])
                
                P_y_given_s = self.prob_P_y_given_s(self, Y, p_js)
                
                if t == 0:
                    P_s_given_Z = self.prob_P_s_given_Z(self, param, shapes, Z)
                    alpha[:,i,t] = np.multiply(P_y_given_s,P_s_given_Z)
                    beta[:,i,v] = np.ones( (n_segments,1,1) )
                else:
                    P_s_given_r = self.prob_P_s_given_r(self, param, shapes, Z)
    
                    sum_alpha = np.zeros( (n_segments) )
                    sum_beta = np.zeros( (n_segments) )
                    for r in range(0,n_segments):
                        sum_alpha = sum_alpha + np.multiply( np.multiply(alpha[:,i,t-1],P_s_given_r[:,r]), P_y_given_s)
                        sum_beta = sum_beta + np.multiply( np.multiply(beta[:,i,v+1],P_s_given_r[r,]), P_y_given_s)
    
                
                alpha[:,i,t] = sum_alpha
                beta[:,i,t]  = sum_beta
                
        return alpha, beta
    
      
     def maximization_step_cov(self, alpha, beta, param_in, shapes, n_segments):
        """function for the maximization step"""
        
        x0 = param_in
        
        """perform the maximization"""
        param_out = self.sci.optimize.minimize(optimization_function_cov, x0, args=(alpha, beta, param_in, shapes, n_segments))
           
        return param_out
    
    
    def optimization_function_cov(self, x, alpha, beta, param_in, shapes, n_segments):
        """function that has to be minimized"""
                
        
        """compute function"""
        sum = 0;
        
        for i in range(0,self.n_customers):   
            
            Y = self.list_Y[0][i,:]
            if self.covariates = True:
                Z = self.list_Z[0][i,:]    
            else: 
                Z = np.array([])
            
            P_s_given_Z = self.prob_P_s_given_Z(x, shapes, Z) 
            P_s_given_Y_Z = self.state_event(alpha,beta,i,0)
            
            sum = sum + np.sum(np.multiply(P_s_given_Y_Z, math.log(P_s_given_Z)))
            
            for t in range(1,self.T):
                Y = self.list_Y[0][i,:]
                if self.covariates = True:
                    Z = self.list_Z[0][i,:]    
                else: 
                    Z = []
             
                P_s_given_r = self.prob_P_s_given_r(x, shapes, Z)
    
                for r in range(0,n_segments):
                    for s in range(0,n_segments):
                        
                        P_sr_given_Y_Z = self.joint_event(Y, Z, alpha, beta, param_in, shapes, i, t, s, r)
                        P_s_given_r = P_s_given_r[s,r]
                        
                        sum = sum + P_sr_given_Y_Z * math.log(P_s_given_r)
    
            for t in range(0,self.T):
                for s in range(0,n_segments):
                    P_s_given_Y_Z = self.state_event(alpha,beta,i,t)
                    P_y_given_s = self.prob_P_y_given_s(Y, self.prob_p_js(x, shapes))
    
                    sum = sum + P_s_given_Y_Z*math.log(P_y_given_s)
    
        return sum    

    

    def prediction(....)
