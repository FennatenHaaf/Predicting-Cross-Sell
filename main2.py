# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:43:27 2021

@author: matth
"""

import numpy as np 
from math import comb
import matplotlib.pyplot as plt
from scipy.stats import multinomial


#initialise 
data = 
Y = 
Z =

n_segments = 
n_covariates = 
n_products = 
n_customers = 
n_categories = 
n_customers = 

T = 


#parameters of P(S_0 = s|Z)
gamma_0 = np.ones( (n_segments-1,n_covariates+1) )

#parameters of P(S_t = s | S_t+1 = r)
gamma_sr_0 = np.ones( (n_segments-1,n_segments) )
gamma_sk_t = np.ones( (n_segments-1,n_covariates) )

#parameters of P(Y|s) \\ kan ook (n_categories,n_segments)!!
beta = ( (n_products,n_categories,n_segments) )


tol = 10^-4 * np.ones(n_parameters,1) #tolerance


#Begin algorithm
def EM(gamma_0, gamma_sr_0, gamma_sk_t, beta, tolerance):
    iteration = 0
    while difference
        
        gamma_0_in = gamma_0_out
        gamma_sr_0_in = gamma_sr_0_out
        gamma_sk_t_in = gamma_sk_t_out
        beta_in = beta_out
        
        [] = expectation_step(gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in)
        
        [gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out] = maximization_step()
        
        difference = (any(abs(gamma_0_in-gamma_0_out)) > tol) | (any(abs(gamma_sr_0_in-gamma_sr_0_out)) > tol) | (any(abs(gamma_sk_t_in-gamma_sk_t_out)) > tol) | (any(abs(beta_in-beta_out > tol)))
        
        iteration = iteration + 1
    return 


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


#van pi naar probabilities P(Y|s)
def prob_P_y_given_s(y, pi):
    n_products = pi.shape(0)
    n_categories = pi.shape(1)
    n_segments = pi.shape(2)

    P_y_given_s = np.ones( (n_segments,1) )
    
    for p in range(0,n_products):
        for c in range(0,n_categories):
            P_y_given_s = P_y_given_s * pi[p,c,:]**(y[p] == c)   
    return P_y_given_s 


#from gamma_0 to probabilities P(S_0 = s|Z)
def prob_P_s_given_Z(gamma_0, Z):  
    n_segments = gamma_0.shape[0]+1
    
    P_s_given_Z = np.zeros((n_segments-1))
    
    P_s_given_Z = exp( gamma_0[:,0] + np.matmul(gamma_0[:,1:n_covariates+1] , Z) )
    P_s_given_Z = np.vstack([P_s_given_Z, [1]]) #for the base case
    P_s_given_Z = P_s_given_Z/np.sum(P_s_given_Z)
    
    return P_s_given_Z
    

#from gamma_sr_0 and gamma_sk_t to probabilities P(S_t = s | S_t+1 = r)
def prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z):
    n_segments = gamma_sk_t.shape[0]+1

    P_s_given_r = np.zeros((n_segments,n_segments))
    
    for r in range (0,n_segments):
        P_s_given_r[:,r] = exp( gamma_sr_0[:,r] + np.matmul(gamma_sk_t[:,0:n_covariates] , Z ) 
        P_s_given_r[:,r] = np.concatenate(P_s_given_r[:,r],[1])#for the base case
        P_s_given_r[:,r] = P_s_given_r[:,r]/np.sum(P_s_given_r)

    return P_s_given_r
    
    











