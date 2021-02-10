# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:39:33 2021

@author: Matthijs van Heese
"""
import extra_functions_HMM as ef
import numpy as np 
import math
import utils
from tqdm import tqdm
from scipy.optimize import minimize 


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
    
    for s in tqdm(range(0,n_segments)):
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
        
    
def prob_P_y_given_s(self, Y, p_js, n_segments):
    """function to compute P(Y_it | X_it = s) with probabilities p"""

    row_Y = len(Y)

    P_y_given_s = np.ones((row_Y, n_segments))
        
    for p in range(0,self.n_products):
        for c in range(0,self.n_categories[p]):
            prob_p_c = np.transpose(np.power(np.transpose([p_js[:,p,c]]), [Y[:,p] == c]))
            P_y_given_s = np.multiply(P_y_given_s, prob_p_c)   
    return P_y_given_s 
    
def prob_P_s_given_Z(self, param, shapes, Z, n_segments):  
    """function to compute P(X_i0 = s| Z_i0) with parameters gamma_0"""
    
    row_Z = len(Z)
    
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self,param,shapes)

        P_s_given_Z = np.exp(np.transpose([gamma_0[:,0]] * row_Z) + np.matmul(gamma_0[:,1:self.n_covariates+1] , np.transpose(Z) ) )
        P_s_given_Z = np.vstack( (P_s_given_Z, np.ones((1,row_Z))) )  #for the base case
        P_s_given_Z = np.transpose(np.divide( P_s_given_Z , np.sum(P_s_given_Z, axis = 0) ))   
    else:
        A, pi, b = param_list_to_matrices(self,param,shapes)
        P_s_given_Z = pi
   
    return P_s_given_Z
        
    
def prob_P_s_given_r(self, param, shapes, Z, n_segments):
    """function to compute P(X_it = s | X_it-1 = r, Z_it)"""
    
    row_Z = len(Z)

    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self,param,shapes)

        gamma_sr_0 = np.vstack((gamma_sr_0, np.zeros((1,n_segments))))
        P_s_given_r = np.repeat(gamma_sr_0[np.newaxis,:,:], row_Z, axis = 0)
        
        mat = np.matmul(gamma_sk_t, np.transpose(Z))
        mat = np.vstack((mat, np.zeros((1,row_Z))))
        mat = np.repeat(mat, n_segments, axis = 1)
        mat = np.array_split(mat, row_Z, axis = 1)
        mat = np.array(mat)
        
        P_s_given_r = np.exp(P_s_given_r + mat)

        P_s_given_r = np.divide(P_s_given_r, np.reshape(np.sum(P_s_given_r,1), (row_Z,1,n_segments)))
        
    else:  
            A, pi, b = self.param_list_to_matrices(param,shapes)
            P_s_given_r = A
        
    return P_s_given_r
        
    

#--------------------function for maximisation step---------------------



def joint_event(self, Y, Z, alpha, beta, param, shapes, t, s, r, n_segments):#for all person
    """function to compute P(X_it-1 = s_t-1, X_it = s_t|Y_i, Z_i)"""
    
    P_s_given_Y_Z = np.multiply(alpha, beta)
    
    P_s_given_r = prob_P_s_given_r(self, param, shapes, Z, n_segments)
    P_y_given_s = prob_P_y_given_s(self, Y, prob_p_js(self, param, shapes, n_segments), n_segments)
    
    P_sr_given_Y_Z = np.multiply(alpha[r,:,t-1], P_s_given_r[:,s,r])
    P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, P_y_given_s[:,s])
    P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, beta[s,:,t])
    P_sr_given_Y_Z = np.divide(P_sr_given_Y_Z, np.sum(P_s_given_Y_Z[:,:,t], axis = 0))
    
    return P_sr_given_Y_Z 

def state_event(self, alpha, beta, n_segments): #alpha/beta = s, i, t
    """function to compute P(X_it = s|Y_i, Z_i)"""

    P_s_given_Y_Z = np.multiply(alpha, beta)
    P_s_given_Y_Z = P_s_given_Y_Z/np.sum(P_s_given_Y_Z, axis = 0)
    
    return P_s_given_Y_Z



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
                
            P_y_given_s = prob_P_y_given_s(self, Y, p_js, n_segments)
                
            if t == 0:
                P_s_given_Z = prob_P_s_given_Z(self, param, shapes, Z, n_segments)
                alpha[:,i,t] = np.multiply(P_y_given_s,P_s_given_Z)
                beta[:,i,v] = np.ones((n_segments))
            else:
                P_s_given_r = prob_P_s_given_r(self, param, shapes, Z, n_segments)
    
                sum_alpha = np.zeros( (n_segments) )
                sum_beta = np.zeros( (n_segments) )
                for r in range(0,n_segments):
                    sum_alpha = sum_alpha + np.multiply( np.multiply(alpha[:,i,t-1],P_s_given_r[:,r]), P_y_given_s)
                    sum_beta = sum_beta + np.multiply( np.multiply(beta[:,i,v+1],P_s_given_r[r,]), P_y_given_s)
                alpha[:,i,t] = sum_alpha
                beta[:,i,v]  = sum_beta
                
    return alpha, beta

def maximization_step(self, alpha, beta, param_in, shapes, n_segments, max_method):
    """function for the maximization step"""
        
    x0 = param_in
        
    """perform the maximization"""
    param_out = minimize(optimization_function, x0, args=(self, alpha, beta, param_in, shapes, n_segments), method=max_method)
           
    return param_out
    

def optimization_function(x, self, alpha, beta, param_in, shapes, n_segments):
    """function that has to be minimized"""
                
        
    """compute function"""
    sum = 0;
        
    P_s_given_Y_Z = state_event(self, alpha, beta, n_segments)

    Y = self.list_Y[0]
    if self.covariates == True:
        Z = self.list_Z[0]
    else: 
        Z = np.array([])
    
    P_s_given_Z = prob_P_s_given_Z(self, x, shapes, Z, n_segments)  #i x s
    P_s_given_Y_Z_0 = np.transpose(P_s_given_Y_Z[:,:,0]) #s x i x t
    sum = sum + np.sum(np.multiply(P_s_given_Y_Z_0, np.log(P_s_given_Z)))
    
    P_s_given_r = prob_P_s_given_r(self, x, shapes, Z, n_segments)

    for t in tqdm(range(1,self.T)):
        Y = self.list_Y[t]
        if self.covariates == True:
            Z = self.list_Z[t]   
        else: 
            Z = []

        for r in range(0,n_segments):
            for s in range(0,n_segments):
                P_sr_given_Y_Z = joint_event(self, Y, Z, alpha, beta, param_in, shapes, t, s, r, n_segments)
                
                sum = sum + np.sum( np.multiply(P_sr_given_Y_Z, np.log(P_s_given_r[:,s,r]))  )
    
    for t in tqdm(range(0,self.T)):
        Y = self.list_Y[t]        
        P_y_given_s = prob_P_y_given_s(self, Y, prob_p_js(self, x, shapes, n_segments), n_segments)
        P_s_given_Y_Z_t = np.transpose(P_s_given_Y_Z[:,:,t])
        
        sum = sum + np.sum(np.multiply(P_s_given_Y_Z_t[s],np.log(P_y_given_s[s])))
    
    return -sum    

   





