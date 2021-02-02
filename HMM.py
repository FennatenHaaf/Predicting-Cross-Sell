# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:43:27 2021

@author: matth
"""

import numpy as np 
from math import comb
import matplotlib.pyplot as plt
import scipy.stats as sci

class HMM:
    
def initialisation(self, list_dataframes, list_dep_var, list_covariates, n_segments, tol):
    
                   self.list_dataframes = list_dataframes
                   self.list_dep_var = list_dep_var
                   self.list_covariates = list_covariates
                   self.n_segments = n_segments
                   self.n_covariates = size(list_covariates)
                   self.n_dep_var = size(list_dep_var)
                   self.n_customers = 
                   self.n_categories = 
                   self.T = size(list_dataframe)
                   self.tol = tol

                   #parameters of P(S_0 = s|Z)
                   gamma_0 = np.ones( (self.n_segments-1,self.n_covariates+1) )
                    
                   #parameters of P(S_t = s | S_t+1 = r)
                   gamma_sr_0 = np.ones( (self.n_segments-1,self.n_segments) )
                   gamma_sk_t = np.ones( (self.n_segments-1,self.n_covariates) )
                    
                   #parameters of P(Y|s) \\ kan ook (n_categories,n_segments)!!
                   beta = ( (self.n_products,self.n_categories,self.n_segments) )
                    


#Begin algorithm
def EM(gamma_0, gamma_sr_0, gamma_sk_t, beta, tolerance):
    iteration = 0
    while difference
        
        gamma_0_in = gamma_0_out
        gamma_sr_0_in = gamma_sr_0_out
        gamma_sk_t_in = gamma_sk_t_out
        beta_in = beta_out
        
        [alpha,beta] = expectation_step(Y, Z, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in)
        
        [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out] = maximization_step(Y, Z, , alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in, T)
        
        difference = (any(abs(gamma_0_in-gamma_0_out)) > tol) | (any(abs(gamma_sr_0_in-gamma_sr_0_out)) > tol) | (any(abs(gamma_sk_t_in-gamma_sk_t_out)) > tol) | (any(abs(beta_in-beta_out > tol)))
        
        iteration = iteration + 1
        
        return [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out]

#------------Functies waarmee je van de parameters naar de kansen gaat------------
    
#van beta naar probabilities pi(j,c,s) // kan ook pi(c,s)
def prob_pi(beta): 
    pi = np.ones( (n_products,n_categories,n_segments) )
    
    for j in range(0,n_products):
        for s in range(0,n_segments):
            denominator = 0
            for c in range(0,n_categories):
                if c == 0: #first category (zero ownership of product) is the base
                        log_odds[j,c,s] = 0
                        denominator = denominator + exp(log_odds[j,c,s])
                else:
                    if s == n_segments - 1:
                        log_odds[j,c,s] = beta[j,c,s]
                        denominator = denominator + exp(log_odds[j,c,s])
                    else: 
                        log_odds[j,c,s] = beta[j,c,n_segments-1] + beta[j,c,s]
                        denominator = denominator + exp(log_odds[j,c,s])
            pi[j,:,s] = pi[j,:,s] / denominator
            
    return pi


#van pi naar probabilities P(Y_it | X_it = s)
def prob_P_y_given_s(y, pi):
    n_products = pi.shape(0)
    n_categories = pi.shape(1)
    n_segments = pi.shape(2)

    P_y_given_s = np.ones( (n_segments,1) )
    
    for p in range(0,n_products):
        for c in range(0,n_categories):
            P_y_given_s = P_y_given_s * pi[p,c,:]**(y[p] == c)   
    return P_y_given_s 


#from gamma_0 to probabilities P(X_i0 = s| Z_i0)
def prob_P_s_given_Z(gamma_0, Z):  
    n_segments = gamma_0.shape[0]+1
    
    P_s_given_Z = np.zeros((n_segments-1))
    
    P_s_given_Z = exp( gamma_0[:,0] + np.matmul(gamma_0[:,1:n_covariates+1] , Z) )
    P_s_given_Z = np.vstack([P_s_given_Z, [1]]) #for the base case
    P_s_given_Z = P_s_given_Z/np.sum(P_s_given_Z)
    
    return P_s_given_Z
    

#from gamma_sr_0 and gamma_sk_t to probabilities P(X_it = s | X_it-1 = r, Z_it)
def prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z):
    n_segments = gamma_sk_t.shape[0]+1

    P_s_given_r = np.zeros((n_segments,n_segments))
    
    for r in range (0,n_segments):
        P_s_given_r[:,r] = exp( gamma_sr_0[:,r] + np.matmul(gamma_sk_t[:,0:n_covariates] , Z ) 
        P_s_given_r[:,r] = np.concatenate(P_s_given_r[:,r],[1])#for the base case
        P_s_given_r[:,r] = P_s_given_r[:,r]/np.sum(P_s_given_r)

    return P_s_given_r
    
    
#------------Functions for the expectation step------------
    
#function for expectation step
def expectation_step(gamma_0, gamma_sr_0, gamma_sk_t, beta, t):
        
    [alpha,beta] = forward_backward_procedure(Y,Z, beta, gamma_0, gamma_sr_0, gamma_sk_t, t)
    

    return

#forward-backward algorithm
def forward_backward_procedure(Y,Z, beta, gamma_0, gamma_sr_0, gamma_sk_t, T):
    
    n_customers = Y.shape[0]
    n_segments = beta.shape[2]
    pi = prob_pi(beta)
    
    alpha = np.zeros(n_customers, T, n_segment)
    beta = np.zeros(n_customers, T, n_segment)

    for i in range(0,n_customers):
        for t in range(0,T):
            v = T - t
  
            P_y_given_s = prob_P_y_given_s(Y[i,:,t], pi)
            
            if t == 0:
                P_s_given_s = prob_P_s_given_Z(gamma_0, Z)
                alpha[i,t,:] = multiply(P_y_given_s,P_s_given_Z)
                beta[i,v,:] = np.ones( (1,1,n_segments) )
            else:
                P_s_given_r = prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z)

                sum_alpha = np.zeros( (n_segments,1) )
                sum_beta = np.zeros( (n_segments,1) )
                for r in range(0,n_segments):
                    sum_alpha = sum_alpha + multiply( multiply(alpha[i,t-1,:],P_s_given_r[:,r]), P_y_given_s)
                    sum_beta = sum_beta + multiply( multiply(beta[i,v+1,:],P_s_given_r[r,]), P_y_given_s)

            
            alpha[i,t,:] = sum_alpha
            beta[i,t,:]  = sum_beta
            
    return [alpha,beta]



#------------Functies voor de maximiation step------------

#function for maximization step
def maximization_step(Y, Z, , alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in, T):
    
    
    n_gamma_0 = gamma_0.size
    n_gamma_sr_0 = gamma_sr_0.size
    n_gamma_sk_t = gamma_sk_t.size
    n_beta = beta.size
    n_parameters = n_gamma_0 + n_gamma_sr_0 + n_gamma_sk_t + n_beta
    
    x0 = concatenate(gamma_0.flatten(), gamma__sr_0.flatten() gamma_sk_t.flatten() beta.flatten())
    
    [param_out] = sci.optimize.minimize(optimization_function, x0, args=(alpha,beta,Y,Z,T))
    
    #get right output
    gamma_0_out = param_out[0:n_gamma_0]
    gamma_sr_0_out = [n_gamma_0:n_gamma_0+n_gamma_sr_0]
    gamma_sk_t_out = [n_gamma_0+n_gamma_sr_0:n_gamma_0+n_gamma_sr_0+n_gamma_sk_t]
    beta_out = [n_gamma_0+n_gamma_sr_0+n_gamma_sk_t:n_parameters]
    
    #transform parameter output to matrix 
    
    return [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out]

#function that has to be minimized
def optimization_function(x, alpha, beta, Y, Z, T):
    
    return 

#function for deriving P(X_it = s|Y)_i, Z_i
def joint_event():
    
    return

#function for deriving P(X_it-1 = s. X_it = s|Y_i, Z_i)
def state_event():
    
    return









