# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:39:33 2021

@author: Matthijs van Heese
"""
import numpy as np 
import math
from tqdm import tqdm
from time import perf_counter
import numexpr as ne


def param_list_to_matrices(self, n_segments, param, shapes):
    """change the shape of the parameters to separate matrices"""

    if self.covariates == True:

        n_gamma_0 = shapes[0,1]
        n_gamma_sr_0  = shapes[1,1]   
        n_gamma_sk_t = shapes[2,1]
        gamma_0 = param[0:n_gamma_0]
        gamma_sr_0 = param[n_gamma_0:(n_gamma_0+n_gamma_sr_0)]
        gamma_sk_t = param[(n_gamma_0+n_gamma_sr_0):(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t)]
        
        gamma_0 = np.reshape(gamma_0, shapes[0,0])
        gamma_sr_0 = np.reshape(gamma_sr_0, shapes[1,0])                        
        gamma_sk_t = np.reshape(gamma_sk_t, shapes[2,0])
        
        beta_param = param[(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t):param.shape[0]]
        beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)

        i = 0
        for s in range(n_segments):
            for p in range(0,self.n_products):
                beta[s,p,0:self.n_categories[p]-1] = beta_param[i:i+self.n_categories[p]-1]
                i = i + self.n_categories[p]-1
        
        return gamma_0, gamma_sr_0, gamma_sk_t, beta
    else:
        n_A = shapes[0,1]
        n_pi = shapes[1,1]
        try:
            A = param[0:n_A]
        except:
            param = param.x
            A = param[0:n_A]
        pi = param[n_A:(n_A+n_pi)]
        A = np.reshape(A, shapes[0,0])
        pi = np.reshape(pi, shapes[1,0])  
        
        b_param = param[(n_A+n_pi):param.shape[0]]
        b = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)
        i = 0
        for s in range(n_segments):
            for p in range(0,self.n_products):
                b[s,p,0:self.n_categories[p]-1] = b_param[i:(i+self.n_categories[p]-1)]
                i = i + self.n_categories[p]-1

        return A, pi, b
        
        
def param_matrices_to_list(self, n_segments, A = [], pi = [], b = [], gamma_0 = [], gamma_sr_0 = [], gamma_sk_t = [], beta = []):
    """transform parameter matrices to one vector"""

    if self.covariates == True:
        
        beta_vec = np.array([0])
        
        for s in range(n_segments):
            for p in range(0,self.n_products):
                beta_vec = np.concatenate( (beta_vec, beta[s,p,0:self.n_categories[p]-1]) )
        beta_vec = beta_vec[1:]
        
        parameters = np.concatenate((gamma_0.flatten(), gamma_sr_0.flatten(), gamma_sk_t.flatten(), beta_vec))
    else:
        b_vec = np.array([0])
        
        for s in range(n_segments):
            for p in range(0,self.n_products):
                b_vec = np.concatenate( (b_vec, b[s,p,0:self.n_categories[p]-1]) )
                
        b_vec = b_vec[1:b_vec.size]
        
        parameters = np.concatenate((A.flatten(), pi.flatten(), b_vec))
    
    return parameters



 #------------Functies waarmee je van de parameters naar de kansen gaat------------
def prob_p_js(self, param, shapes, n_segments): 
    """function to compute p(j,c,s) with parameters beta/b"""
        
    p_js = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    log_odds = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)
    else:
        A, pi, b = param_list_to_matrices(self, n_segments, param, shapes)
        beta = b
    
    for s in range(0,n_segments):
        for p in range(0,self.n_products):
            for c in range(0,self.n_categories[p]):
                if c == 0: #first category (zero ownership of product) is the base
                    log_odds[s,p,c] = 0
                else:
                    if s == (n_segments-1): #last segment is the base
                        log_odds[s,p,c] = beta[s,p,(c-1)]
                    else: 
                        log_odds[s,p,c] = beta[(n_segments-1),p,(c-1)] + beta[s,p,(c-1)]
            p_js[s,p,0:self.n_categories[p]] = np.exp(log_odds[s,p,:self.n_categories[p]] - logsumexp(log_odds[s,p,:self.n_categories[p]]))
                        
    return p_js
        

def prob_P_y_given_s(self, Y, p_js, n_segments):
    """
    Function to compute P(Y_it | X_it = s)  with probabilities p.
    This is done by calculating P(Y_ijt| X_it) for every j

    """
    row_Y = len(Y)

    P_y_given_s = np.ones((row_Y, n_segments))
    for j in range(0,self.n_products):
        for c in range(0,self.n_categories[j]):
            prob_j_c = np.power(p_js[:, j, c][np.newaxis, :], Y[:, j][:, np.newaxis] == c)
            P_y_given_s = np.multiply(P_y_given_s, prob_j_c)

    return P_y_given_s

def prob_P_s_given_Z(self, param, shapes, Z, n_segments):   
    """function to compute P(X_i0 = s| Z_i0) with parameters gamma_0"""
    
    row_Z = len(Z)
    
    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)
        P_s_given_Z = gamma_0[:,0][:,np.newaxis] + gamma_0[:,1:].dot(np.transpose(Z)) 
        P_s_given_Z = np.vstack( (P_s_given_Z, np.zeros((1,row_Z))) )  #for the base case
        
        P_s_given_Z = np.transpose( np.exp( P_s_given_Z - logsumexp(P_s_given_Z) ))   

    else:
        A, pi, b = param_list_to_matrices(self, n_segments, param, shapes)
        pi = np.append(pi, np.array([1]))
        P_s_given_Z = np.exp( pi - logsumexp(pi) )
   
    return P_s_given_Z
        
    
def prob_P_s_given_r(self, param, shapes, Z, n_segments):
    """function to compute P(X_it = s | X_it-1 = r, Z_it)"""
    
    row_Z = len(Z)

    """case with covariates"""
    if self.covariates == True:
        gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)

        gamma_sr_0 = np.vstack((gamma_sr_0, np.zeros((1,n_segments))))
        P_s_given_r = np.repeat(gamma_sr_0[np.newaxis,:,:], row_Z, axis = 0)
        
        mat = np.matmul(gamma_sk_t, np.transpose(Z)) 
        mat = np.vstack((mat, np.zeros((1,row_Z))))
        mat = np.repeat(mat, n_segments, axis = 1)
        mat = np.array_split(mat, row_Z, axis = 1)
        mat = np.array(mat)
        
        P_s_given_r = P_s_given_r + mat

        #P_s_given_r = np.divide(P_s_given_r, np.reshape(np.sum(P_s_given_r,1), (row_Z,1,n_segments)))
        log_sum_exp = logsumexp(P_s_given_r, axis = 1, reshape = True)
        P_s_given_r = np.exp(P_s_given_r - log_sum_exp[:,np.newaxis,:]) #i x s x s
    else:  
        A, pi, b = param_list_to_matrices(self, n_segments, param, shapes)
        A = np.vstack((A, np.zeros((1, n_segments))))
        
        A = A[np.newaxis, :, :]
        P_s_given_r = np.exp(A - logsumexp(A))
    return P_s_given_r
        
    def gamma_sr_0_to_trans(self, param, shapes, n_segments):
        if self.covariates == True:
            gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)
            gamma_sr_0 = np.vstack((gamma_sr_0, np.zeros((1,n_segments))))
            
            P_s_given_r = np.exp(gamma_sr_0 - logsumexp(gamma_sr_0)) #i x s x s
            
        else:
            A, pi, b = param_list_to_matrices(self, n_segments, param, shapes)
            A = np.vstack((A, np.zeros((1, n_segments))))
        
            P_s_given_r = np.exp(A - logsumexp(A))
         
        return P_s_given_r
            

#--------------------function for maximisation step---------------------


def joint_event(self, alpha, beta, t, n_segments,
                P_s_given_Y_Z_ut, P_s_given_r, P_y_given_s):#for all person
    """function to compute P(X_it-1 = s_t-1, X_it = s_t|Y_i, Z_i)"""
    
    #P_s_given_Y_Z_ut = np.multiply(alpha, beta)
    #P_s_given_r = prob_P_s_given_r(self, param, shapes, Z, n_segments)
    #P_y_given_s = prob_P_y_given_s(self, Y, prob_p_js(self, param, shapes, n_segments), n_segments)
    
    #P_sr_given_Y_Z = np.multiply(alpha[r,:,t-1], P_s_given_r[:,s,r])  #sxixt ixsxs  
    #P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, P_y_given_s[:,s])
    #P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, beta[s,:,t])
    #P_sr_given_Y_Z = np.divide(P_sr_given_Y_Z, np.sum(P_s_given_Y_Z[:,:,t], axis = 0))
    
    #P_sr_given_Y_Z = np.multiply(np.transpose(alpha[:,:,t-1]), P_s_given_r[:,s,:])  #[sxi]' [ixs] = [ixs]  
    #P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, np.transpose([P_y_given_s[:,s]])) #[ixs] [ixs] = [ixs]
    #P_sr_given_Y_Z = np.multiply(P_sr_given_Y_Z, np.transpose([beta[s,:,t]])) # [ixs] [sxi]' = [ixs]
    #P_sr_given_Y_Z = np.divide(P_sr_given_Y_Z, np.transpose([np.sum(P_s_given_Y_Z_ut[:,:,t], axis = 0)])) 
    
    #new:
        
    P_sr_given_Y_Z = np.zeros((self.n_customers, n_segments, n_segments))
    for s in range(0,n_segments):
        mat = np.multiply(np.transpose(alpha[:,:,t-1]), P_s_given_r[:,s,:])  #[sxi]' [ixs] = [ixs]  
        mat = np.multiply(mat, np.transpose([P_y_given_s[:,s]])) #[ixs] [ixs] = [ixs]
        mat = np.multiply(mat, np.transpose([beta[s,:,t]])) # [ixs] [sxi]' = [ixr]
        P_sr_given_Y_Z[:,s,:] = mat
    
    sum_per_cust = np.sum(np.sum(P_sr_given_Y_Z, axis = 2), axis = 1)
    P_sr_given_Y_Z = np.divide(P_sr_given_Y_Z, sum_per_cust[:,np.newaxis, np.newaxis])
      
        
    return P_sr_given_Y_Z 


def state_event(self, alpha, beta): #alpha/beta = s, i, t
    """function to compute P(X_it = s|Y_i, Z_i)"""

    P_s_given_Y_Z = np.multiply(alpha, beta)  #s x i x t
    P_Y_given_Z = np.sum(P_s_given_Y_Z, axis = 0) 
    P_s_given_Y_Z = np.divide(P_s_given_Y_Z, P_Y_given_Z)
    
    return P_s_given_Y_Z


def logsumexp(x, axis = 0, row_Z = 1, reshape = False):
    
    c = x.max(axis = axis)
    
    if reshape == False:
        diff = x - c 
    else:
        diff = x - c[:,np.newaxis,:]
        
    return c + np.log(np.sum(np.exp(diff), axis = axis))









