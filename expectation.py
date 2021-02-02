# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:08:14 2021

@author: matth
"""
import numpy as np 
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

#function for expectation step
def expectation_step(gamma_0, gamma_sr_0, gamma_sk_t, beta, t):
        
    [alpha,beta] = forward_backward_procedure(Y,Z, beta, gamma_0, gamma_sr_0, gamma_sk_t, t)
    
    
    [] = joint_event()
    [] = state_event()
    return

#subfunctions for expectation step
def forward_backward_procedure(Y,Z, beta, gamma_0, gamma_sr_0, gamma_sk_t, T):
    
    n_segments = gamma_sr_0.shape[0]

    pi = prob_pi(beta)
    
    alpha = np.zeros(n_customers, T, n_segment)
    beta = np.zeros(n_customers, T, n_segment)


    for i in range(0,n_customers):
        for t in range(0,T):
            v = T - t
            
            #initialise P(S_0 = s|Z) and P(S_t = s | S_t+1 = r)
            if t == 0:
                P_s_given_Z = np.zeros(n_segments,1)
            else:
                P_s_given_r = np.zeros(n_segments,n_segments)

            #make P(S_0 = s|Z) and P(S_t = s | S_t+1 = r)
            if t == 0:
                P_s_given_Z[:,1] = exp( gamma_0[:,0] + gamma_0[:,[1:n_covariates]] * Z[i,:,t] )
            else:
                for r in range (0,n_segments):
                    P_s_given_r[:,r] = exp( gamma_sr_0[:,r] + gamma_sk_t[:,[1:n_covariates]] * Z[i,:,t] ) 
            

            P_s_given_Z = P_s_given_Z/sum(P_s_given_Z)
                
            for s in range(0,n_segments):
                P_s_given_r[:,s] = P_s_given_r[:,s]/sum(P_s_given_r[:,s])

  
            sum_alpha = 0
            sum_beta = 0
            P_y_given_s = multinomial(Y[i,:,t], pi, n_segments)
            
            if t == 0:
                alpha[i,t,:] = multiply(P_y_given_s,P_s_given_Z)
                beta[i,v,:] = np.ones( (n_segments,1) )
            else:
                sum_alpha = np.zeros( (n_segments,1) )
                sum_beta = np.zeros( (n_segments,1) )
                for r in range(0,n_segments):
                    sum_alpha = sum_alpha + multiply( multiply(alpha[i,t-1,:],P_s_given_r[:,r]), P_y_given_s)
                    sum_beta = sum_beta + multiply( multiply(beta[i,v+1,:],P_s_given_r[r,]), P_y_given_s)

            
            alpha[i,t,:] = sum_alpha
            beta[i,t,:]  = sum_beta
                
        
               
    return [alpha,beta]


    
def multinomial(y, pi, n_segments):
    n_products = pi.shape(1)
    n_categories = pi.shape(2)
    
    product = np.ones( (n_segments,1) )
    for p in range(0,n_products):
        for c in range(0,n_categories):
            product = product * pi[p,c,:]**(y[p] == c)   
    return product 


    
def joint_event():
    
    return

def state_event():
    
    return
