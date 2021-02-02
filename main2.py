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


#parameters of pi
gamma_0 = ones( (n_segments-1,n_covariates+1) )

#parameters of A
gamma_sr_0 = ones( (n_segments,n_segments) )
gamma_sk_t = ones( (n_segments-1,n_covariates) )

#parameters of B \\ kan ook (n_categories,n_segments)!!
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


    


#from beta to probabilities pi(j,c,s) // kan ook pi(c,s)
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

         
               
    
    











