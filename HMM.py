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
           list_covariates: list consisting of all the names of the variables we use as covariates"""
           
           
        self.list_dataframes = list_dataframes
        self.list_dep_var = list_dep_var
        self.n_dep_var = len(list_dep_var)
        self.list_covariates = list_covariates
        self.n_covariates = len(list_covariates)
        self.T = len(list_dataframes)
        self.n_customers = self.list_dataframes[0].shape[0]
        
        
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


        """HMM with covariates, following notation of Paas (2008)"""
        if self.covariates:
        
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
            
            iteration = 0
            difference = 1
            
            """Start EM procedure"""
            while difference:
                
                gamma_0_in = gamma_0_out
                gamma_sr_0_in = gamma_sr_0_out
                gamma_sk_t_in = gamma_sk_t_out
                beta_in = beta_out
                
                alpha,beta = self.forward_back_procedure_cov(beta_in, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in)
                
                gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out = self.maximization_step_cov(alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in)
                
                iteration = iteration + 1
                            
                difference = (any(abs(gamma_0_in-gamma_0_out)) > tolerance) | (any(abs(gamma_sr_0_in-gamma_sr_0_out)) > tolerance) | (any(abs(gamma_sk_t_in-gamma_sk_t_out)) > tolerance) | (any(abs(beta_in-beta_out) > tolerance))
            
            return gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out
        
        
        
        """HMM without covariates, following more general forward-backward algorithm notation"""
        else:
            
            """Construct parameters to estimate"""
            A = 1/n_segments * np.ones((n_segments,n_segments)) #P(Y_it | X_it = s)
            pi = 1/n_segments * np.ones((n_segments))  #P(X_i0 = s| Z_i0)
            b = np.ones((n_segments, self.n_products, max(self.n_categories))) #parameters for P(Y_it | X_it = s)
            
            iteration = 0
            difference = 1

            """Initialise parameters to estimate"""
            A_out = A
            pi_out = pi
            b_out = b
            
            """Start EM procedure"""
            while difference:
                
                A_in = A_out
                pi_in = pi_out
                b_in = b_out
                
                alpha,beta = self.forward_backward_procedure(A_in, pi_in, b_in)
                
                [A_out, pi_out, b_out] = self.maximization_step(alpha, beta, A_in, pi_in, b_in)
                
                iteration = iteration + 1
                            
                difference = (any(abs(A_in-A_out)) > tolerance) | (any(abs(pi_in-pi_out)) > tolerance) | (any(abs(b_in-bi_out)) > tolerance)
            
            return A_out, pi_out, b_in
        
            
        
    #------------Function for the expectation step for procedure with covariates------------
        
    
    #forward-backward algorithm
    def forward_backward_procedure_cov(self, beta, gamma_0, gamma_sr_0, gamma_sk_t):
        """function for the expectation step:compute alpha and beta with all parameters"""

        n_segments = beta.shape[0]
        pi = self.prob_pi(beta)
        
        alpha = np.zeros(n_segments, self.n_customers, self.T)
        beta = np.zeros(n_segments, self.n_customers, self.T)
    
        for i in range(0,self.n_customers):
            for t in range(0,self.T):
                v = self.T - t
                
                Y = self.list_Y[t][i,:]
                Z = self.list_Z[t][i,:]
                
                P_y_given_s = self.prob_P_y_given_s(self, Y, pi)
                
                if t == 0:
                    P_s_given_Z = self.prob_P_s_given_Z(self,gamma_0, Z)
                    alpha[:,i,t] = np.multiply(P_y_given_s,P_s_given_Z)
                    beta[:,i,v] = np.ones( (n_segments,1,1) )
                else:
                    P_s_given_r = self.prob_P_s_given_r(self, gamma_sr_0, gamma_sk_t, Z)
    
                    sum_alpha = np.zeros( (n_segments) )
                    sum_beta = np.zeros( (n_segments) )
                    for r in range(0,n_segments):
                        sum_alpha = sum_alpha + np.multiply( np.multiply(alpha[:,i,t-1],P_s_given_r[:,r]), P_y_given_s)
                        sum_beta = sum_beta + np.multiply( np.multiply(beta[:,i,v+1],P_s_given_r[r,]), P_y_given_s)
    
                
                alpha[:,i,t] = sum_alpha
                beta[:,i,t]  = sum_beta
                
        return [alpha,beta]
    
      
    def maximization_step_cov(self, Y, Z, alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in):
        """function for the maximization step"""
        
        n_gamma_0 = gamma_0_in.size
        n_gamma_sr_0 = gamma_sr_0_in.size
        n_gamma_sk_t = gamma_sk_t_in.size
        n_beta = beta_in.size
        n_parameters = n_gamma_0 + n_gamma_sr_0 + n_gamma_sk_t + n_beta
        
        x0 = np.concatenate((gamma_0_in.flatten(), gamma_sr_0_in.flatten(), gamma_sk_t_in.flatten(), beta_in.flatten()))
        
        param_out = sci.optimize.minimize(optimization_function, x0, args=(alpha,beta,Y,Z))
        
        #get right output
        gamma_0_out = param_out[0:n_gamma_0]
        gamma_sr_0_out = param_out[n_gamma_0:n_gamma_0+n_gamma_sr_0]
        gamma_sk_t_out = param_out[n_gamma_0+n_gamma_sr_0:n_gamma_0+n_gamma_sr_0+n_gamma_sk_t]
        beta_out = param_out[n_gamma_0+n_gamma_sr_0+n_gamma_sk_t:n_parameters]
        
        #transform parameter output to matrix 
        
        return [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out]
    
    #function that has to be minimized
    def optimization_function(x, alpha, beta, Y, Z, T):
        """function that has to be minimized"""
       
        return 
    
    #function for deriving P(X_it = s|Y_i, Z_i)
    def joint_event():
        
        return
    
    #function for deriving P(X_it-1 = s. X_it = s|Y_i, Z_i)
    def state_event():
        
        return
    
    
    #--------Functions for EM algorithm without covariates----------
     
    
    def forward_backward_procedure(self, A_in, pi_in, b_in)
        """function for the expectation step:compute alpha and beta with all parameters"""

        n_segments = beta.shape[0]
        p = self.prob_pi(b_in)
        
        alpha = np.zeros(n_segments, self.n_customers, self.T)
        beta = np.zeros(n_segments, self.n_customers, self.T)
    
        for i in range(0,self.n_customers):
            for t in range(0,self.T):
                v = self.T - t
                
                Y = self.list_Y[t][i,:]
                Z = self.list_Z[t][i,:]
                
                P_y_given_s = self.prob_P_y_given_s(self, Y, p)
                
                if t == 0:
                    P_s_given_Z = self.prob_P_s_given_Z(self,gamma_0, Z)
                    alpha[:,i,t] = np.multiply(pi_in,P_y_given_s)
                    beta[:,i,v] = np.ones( (n_segments,1,1) )
                else:
    
                    sum_alpha = np.zeros( (n_segments) )
                    sum_beta = np.zeros( (n_segments) )
                    for r in range(0,n_segments):
                        sum_alpha = sum_alpha + np.multiply( np.multiply(alpha[:,i,t-1],A_in[:,r]), P_y_given_s)
                        sum_beta = sum_beta + np.multiply( np.multiply(beta[:,i,v+1],P_s_given_r[r,:]), P_y_given_s)
    
                
                alpha[:,i,t] = sum_alpha
                beta[:,i,v]  = sum_beta
                
        return [alpha,beta]
    
    
    def maximization_step(self, Y, Z, alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in):
        """function for the maximization step"""
        
        
        n_gamma_0 = gamma_0_in.size
        n_gamma_sr_0 = gamma_sr_0_in.size
        n_gamma_sk_t = gamma_sk_t_in.size
        n_beta = beta_in.size
        n_parameters = n_gamma_0 + n_gamma_sr_0 + n_gamma_sk_t + n_beta
        
        x0 = np.concatenate((gamma_0_in.flatten(), gamma_sr_0_in.flatten(), gamma_sk_t_in.flatten(), beta_in.flatten()))
        
        param_out = sci.optimize.minimize(optimization_function, x0, args=(alpha,beta,Y,Z))
        
        #get right output
        gamma_0_out = param_out[0:n_gamma_0]
        gamma_sr_0_out = param_out[n_gamma_0:n_gamma_0+n_gamma_sr_0]
        gamma_sk_t_out = param_out[n_gamma_0+n_gamma_sr_0:n_gamma_0+n_gamma_sr_0+n_gamma_sk_t]
        beta_out = param_out[n_gamma_0+n_gamma_sr_0+n_gamma_sk_t:n_parameters]
        
        #transform parameter output to matrix 
        
        return [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out]
    
    def optimization_function(x, alpha, beta, Y, Z, T):
        """function that has to be minimized"""

        return 
    
    #function for deriving P(X_it = s|Y)_i, Z_i
    def joint_event():
        
        return
    
    #function for deriving P(X_it-1 = s. X_it = s|Y_i, Z_i)
    def state_event():
        
        return
    
    







    #------------Functies waarmee je van de parameters naar de kansen gaat------------
        
    def prob_pi(self,beta): 
        """function to compute pi(j,c,s) with parameters beta"""
        #pi_per_segment = np.ones((self.n_categories[0]))
        #for i in range(1,self.n_products):
        #    pi_per_product = np.zeros((self.n_categories[i]))
        #    pi_per_segment = np.concatenate(pi_per_segment, pi_per_product)
        
        #pi = pi_per_segment
        #for i in range(0,n_segments-1):
        #    pi = np.dstack((pi,pi_per_segment))
        
        n_segments = beta.shape[0]
        pi = np.zeros((n_segments, self.n_products, max(self.n_categories)))
        log_odds = np.zeros((n_segments,self.n_product,max(self.n_categories)))
        
        for p in range(0,self.n_products):
            for s in range(0,n_segments):
                denominator = 0
                for c in range(0,self.n_categories[p]):
                    if c == 0: #first category (zero ownership of product) is the base
                            log_odds[s,p,c] = 0
                            denominator = denominator + math.exp(log_odds[s,p,c])
                    else:
                        if s == n_segments - 1: #last segment is the base
                            log_odds[s,p,c] = beta[s,p,c]
                            denominator = denominator + math.exp(log_odds[s,p,c])
                        else: 
                            log_odds[s,p,c] = beta[n_segments-1,p,c] + beta[s,p,c]
                            denominator = denominator + math.exp(log_odds[s,p,c])
                pi[s,p,:] = pi[s,p,:] / denominator
                
        return pi
    
    
    def prob_P_y_given_s(self, y, pi):
        """function to compute P(Y_it | X_it = s) with probabilities pi"""
        n_segments = pi.shape[0]
        P_y_given_s = np.ones( (n_segments,1) )
        
        for p in range(0,self.n_products):
            for c in range(0,self.n_categories[p]):
                P_y_given_s = P_y_given_s * pi[:,p,c]**(y[p] == c)   
        return P_y_given_s 
    
    
    def prob_P_s_given_Z(self,gamma_0, Z):  
        """function to compute P(X_i0 = s| Z_i0) with parameters gamma_0"""
        n_segments = gamma_0.shape[0]+1
        P_s_given_Z = np.zeros((n_segments-1))
        
        P_s_given_Z = math.exp( gamma_0[:,0] + np.matmul(gamma_0[:,1:self.n_covariates+1] , Z) )
        P_s_given_Z = np.vstack([P_s_given_Z, [1]]) #for the base case
        P_s_given_Z = P_s_given_Z/np.sum(P_s_given_Z)
        
        return P_s_given_Z
        
    
    def prob_P_s_given_r(self, gamma_sr_0, gamma_sk_t, Z):
        """function to compute P(X_it = s | X_it-1 = r, Z_it) with parameters gamma_sr_0 and gamma_sk_t"""
        n_segments = gamma_sk_t.shape[0]+1
        P_s_given_r = np.zeros((n_segments,n_segments))
        for r in range (0,n_segments):
            P_s_given_r[:,r] = math.exp( gamma_sr_0[:,r] + np.matmul(gamma_sk_t[:,0:self.n_covariates], Z) )
            P_s_given_r[:,r] = np.concatenate((P_s_given_r[:,r],[1])) #for the base case
            P_s_given_r[:,r] = P_s_given_r[:,r]/np.sum(P_s_given_r[:,r])
        return P_s_given_r
        
    

    


