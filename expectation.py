# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:08:14 2021

@author: matth
"""
import numpy as np 


#function for expectation step
def expectation_step(Y, Z, gamma_0, gamma_sr_0, gamma_sk_t, beta, T):
        
    [alpha,beta] = forward_backward_procedure(Y,Z, beta, gamma_0, gamma_sr_0, gamma_sk_t, T)
    
    return [alpha,beta]

#subfunctions for expectation step
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


    
