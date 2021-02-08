# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:39:33 2021

@author: Matthijs van Heese
"""

import numpy as np 
import math

def param_list_to_matrices(self,param,shapes):
    """change the shape of the parameters to separate matrices"""

    if self.covariates == True:
        n_gamma_0 = shapes[0,1]
        n_gamma_sr_0  = shapes[1,1]   
        n_gamma_sk_t = shapes[2,1]
        gamma_0 = param[0:n_gamma_0]
        gamma_sr_0 = param[n_gamma_0:(n_gamma_0+n_gamma_sr_0)]
        gamma_sk_t = param[(n_gamma_0+n_gamma_sr_0):(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t)]
        beta = param[(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t):param.shape[0]]
        gamma_0 = np.reshape(gamma_0, shapes[0,0])
        gamma_sr_0 = np.reshape(gamma_sr_0, shapes[1,0])                        
        gamma_sk_t = np.reshape(gamma_sk_t, shapes[2,0])                        
        beta = np.reshape(beta, shapes[3,0])
        return gamma_0, gamma_sr_0, gamma_sk_t, beta
    else:
        n_A = shapes[0,1]
        n_pi = shapes[1,1] 
        A = param[0:n_A]
        pi = param[n_A:(n_A+n_pi)]
        b = param[(n_A+n_pi):param.shape[0]]
        A = np.reshape(A, shapes[0,0])
        pi = np.reshape(pi, shapes[1,0])                      
        b = np.reshape(b, shapes[2,0])       
        return A, pi, b
        
        
def param_matrices_to_list(self, A = [], pi = [], b = [], gamma_0 = [], gamma_sr_0 = [], gamma_sk_t = [], beta = []):
    """transform parameter matrices to one vector"""

    if self.covariates == True:
        parameters = np.concatenate((gamma_0.flatten(), gamma_sr_0.flatten(), gamma_sk_t.flatten(), beta.flatten()))
    else:
        parameters = np.concatenate((A.flatten(), pi.flatten(), b.flatten()))
    
    return parameters




 #------------Functies waarmee je van de parameters naar de kansen gaat------------
        
def prob_p_js(self, param, shapes, n_segments): 
    """function to compute p(j,c,s) with parameters beta/b"""
        
    p_js = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    log_odds = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self,param,shapes)
    else:
        A, pi, b = param_list_to_matrices(self,param,shapes)
        beta = b
    
    for s in range(0,n_segments):
        for p in range(0,self.n_products):
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
            p_js[s,p,0:self.n_categories[p]] = np.exp(log_odds[s,p,0:self.n_categories[p]]) / denominator
                        
    return p_js
        
    
def prob_P_y_given_s(self, y, p_js, n_segments):
    """function to compute P(Y_it | X_it = s) with probabilities p"""

    P_y_given_s = np.ones((n_segments))
        
    for p in range(0,self.n_products):
        for c in range(0,self.n_categories[p]):
            P_y_given_s = np.multiply(P_y_given_s, np.power(p_js[:,p,c], y[p] == c))   
    return P_y_given_s 
    
    
def prob_P_s_given_Z(self, param, shapes, Z, n_segments):  
    """function to compute P(X_i0 = s| Z_i0) with parameters gamma_0"""
        
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self,param,shapes)
        P_s_given_Z = np.exp( gamma_0[:,0] + np.matmul(gamma_0[:,1:self.n_covariates+1] , Z) )
        P_s_given_Z = np.hstack( (P_s_given_Z, np.ones((1,))) ) #for the base case
        P_s_given_Z = P_s_given_Z / np.sum(P_s_given_Z)      
    else:
        A, pi, b = param_list_to_matrices(self,param,shapes)
        P_s_given_Z = pi
   
    return P_s_given_Z
        
    
def prob_P_s_given_r(self, param, shapes, Z, n_segments):
    """function to compute P(X_it = s | X_it-1 = r, Z_it)"""
    
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self,param,shapes)
        P_s_given_r = np.exp(gamma_sr_0 + np.transpose([np.matmul(gamma_sk_t, Z)] * n_segments) )
        P_s_given_r = np.vstack(( P_s_given_r,np.ones((1,n_segments)) )) #for the base case
        P_s_given_r = np.divide(P_s_given_r, np.sum(P_s_given_r,0))
    else:  
            A, pi, b = self.param_list_to_matrices(param,shapes)
            P_s_given_r = A
        
    return P_s_given_r
        
    

#--------------------function for maximisation step---------------------



def joint_event(self, y, z, alpha, beta, param, shapes, i, t, s, r, n_segments):
    """function to compute P(X_it-1 = s_t-1, X_it = s_t|Y_i, Z_i)"""
    
    P_s_given_Y_Z = np.zeros((n_segments))
    P_s_given_Y_Z = np.multiply(alpha[:,i,t], beta[:,i,t])
    
    P_s_given_r = prob_P_s_given_r(self, param, shapes, z, n_segments)
    P_y_given_s = prob_P_y_given_s(self, y, prob_p_js(self, param, shapes, n_segments), n_segments)

    P_sr_given_Y_Z = ( alpha[r,i,t-1] * P_s_given_r[s,r] * P_y_given_s[s] * beta[s,i,t] ) / np.sum(P_s_given_Y_Z)
    
    return P_sr_given_Y_Z

def state_event(self, alpha, beta, i, t, n_segments):
    """function to compute P(X_it = s|Y_i, Z_i)"""

    P_s_given_Y_Z = np.zeros((n_segments))
    P_s_given_Y_Z = np.multiply(alpha[:,i,t], beta[:,i,t])
    P_s_given_Y_Z = P_s_given_Y_Z/np.sum(P_s_given_Y_Z)
    
    return P_s_given_Y_Z













