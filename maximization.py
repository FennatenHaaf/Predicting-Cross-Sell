# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:05:56 2021

@author: matth
"""


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

def optimization_function(x, alpha, beta, Y, Z, T):
    
    return 
    
def joint_event():
    
    return

def state_event():
    
    return
