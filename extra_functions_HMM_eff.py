"""
Contains additional functions to calculate parameters for the Hidden markov Model

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import numpy as np 
import math
from tqdm import tqdm
from time import perf_counter
import numexpr as ne


def param_list_to_matrices(self, n_segments, param, shapes):
    """
    Parameters
    ----------
    n_segments : int
            number of segments being used for the estimation of the HMM
    param : 1D array
            parameters of the HMM
    shapes : 2D array
            array consisting of shape and size of every single parameter matrix
    Returns
    -------
    gamma_0 : 2D array
        values of the parameters for the initialisation probabilities P(S_0 = s|Z)
    gamma_sr_0 : 2D array
        values of the intercept parameters for the transition probabilities P(S_t = s | S_t-1 = r)
    gamma_sr_0 : 2D array
        values of the covariates coefficient parameters for the transition probabilities P(S_t = s | S_t-1 = r)
    beta : 3D array
        values of the parameters for the ownership probabilities P(Y| S_t = s)
    """
    """function for converting the parameters 1D array to seperate parameter matrices """ 
        
    #get shapes of matrices
    n_gamma_0 = shapes[0,1]
    n_gamma_sr_0  = shapes[1,1]   
    n_gamma_sk_t = shapes[2,1]
    
    #get parameters that belong to specific matrices and reshape
    gamma_0 = param[0:n_gamma_0]
    gamma_sr_0 = param[n_gamma_0:(n_gamma_0+n_gamma_sr_0)]
    gamma_sk_t = param[(n_gamma_0+n_gamma_sr_0):(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t)]
        
    gamma_0 = np.reshape(gamma_0, shapes[0,0])
    gamma_sr_0 = np.reshape(gamma_sr_0, shapes[1,0])                        
    gamma_sk_t = np.reshape(gamma_sk_t, shapes[2,0])
        
    #get parameters of beta and fill in the beta matrix. 
    #Number of beta parameters is not equal to the number of elements in beta matrix, for this a for loop is used. 
    beta_param = param[(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t):param.shape[0]]
    beta = np.zeros((n_segments, self.n_products, max(self.n_categories)-1)) #parameters for P(Y| S_t = s)

    i = 0
    for s in range(n_segments):
        for p in range(0,self.n_products):
            beta[s,p,0:self.n_categories[p]-1] = beta_param[i:i+self.n_categories[p]-1]
            i = i + self.n_categories[p]-1
        
    return gamma_0, gamma_sr_0, gamma_sk_t, beta
        
        
def param_matrices_to_list(self, n_segments, gamma_0, gamma_sr_0, gamma_sk_t, beta):
    """
    Parameters
    ----------
    n_segments : int
            number of segments being used for the estimation of the HMM
    gamma_0 : 2D array
        values of the parameters for the initialisation probabilities P(S_0 = s|Z)
    gamma_sr_0 : 2D array
        values of the intercept parameters for the transition probabilities P(S_t = s | S_t-1 = r)
    gamma_sr_0 : 2D array
        values of the covariates coefficient parameters for the transition probabilities P(S_t = s | S_t-1 = r)
    beta : 3D array
        values of the parameters for the ownership probabilities P(Y| S_t = s)
    Returns
    -------
    param : 1D array
            parameters of the HMM
    """
    """function for converting the seperate parameter matrices to one parameter 1D array""" 
        
    #initialise beta list
    beta_vec = np.array([0])
        
    #add beta parameters in beta matrix to beta list
    for s in range(n_segments):
        for p in range(0,self.n_products):
            beta_vec = np.concatenate( (beta_vec, beta[s,p,0:self.n_categories[p]-1]) )
    #the first element was a zero due to initialisation, so remove it
    beta_vec = beta_vec[1:]
    
    #add all flattened other parameters matrices to the beta vector
    parameters = np.concatenate((gamma_0.flatten(), gamma_sr_0.flatten(), gamma_sk_t.flatten(), beta_vec))

    return parameters



# =============================================================================
# Functies waarmee je van de parameters naar de kansen gaat------------
# =============================================================================

def prob_p_js(self, param, shapes, n_segments): 
    """
    Parameters
    ----------
    param : 1D array
            parameters of the HMM
    shapes : 2D array
            array consisting of shape and size of every single parameter matrix
    n_segments : int
            number of segments being used for the estimation of the HMM
    Returns
    -------
    p_js : 3D array
        probabilities P(Y_{pcs}| S_t = s), with segments on the first index, products on the second and category on the third 
    """
    """function to compute p(j,c,s) with parameters beta/b"""
        
    #intialise p_js matrix and logodds matrix
    p_js = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    log_odds = np.zeros((n_segments, self.n_products, max(self.n_categories)))
    
    #get the parameters in matrices instead of one vector
    gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)

    for s in range(0,n_segments):
        for p in range(0,self.n_products):
            for c in range(0,self.n_categories[p]):
                #make for every segment, every product and every category the logodds term
                if c == 0: #first category (zero ownership of product) is the base
                    log_odds[s,p,c] = 0
                else:
                    if s == (n_segments-1): #last segment is the base
                        log_odds[s,p,c] = beta[s,p,(c-1)]
                    else: 
                        log_odds[s,p,c] = beta[(n_segments-1),p,(c-1)] + beta[s,p,(c-1)]
            #transform the logodds values to real probabilties p_js
            p_js[s,p,0:self.n_categories[p]] = np.exp(log_odds[s,p,:self.n_categories[p]] - logsumexp(log_odds[s,p,:self.n_categories[p]]))
                        
    return p_js
        

def prob_P_y_given_s(self, Y, p_js, n_segments):
    """
    Parameters
    ----------
    Y : 2D array
        dependent variables of the HMM algorithm
    p_js : 3D array
        probabilities P(Y_{pcs}| S_t = s), with segments on the first index, products on the second and category on the third 
    n_segments : int
        number of segments being used for the estimation of the HMM
    Returns
    -------
    P_y_given_s : 2D array
        probabilities P(Y_{it} | X_{it} = s), with rows representing customers and columns representing segments 
    """
    """function for computing probabilities P(Y_it | X_it = s) with probabilities p_js"""
        
    #get number of customers for which P(Y_{pcs}| S_t = s) is calculated
    row_Y = len(Y)
    
    #initialise P(Y_{pcs}| S_t = s)
    P_y_given_s = np.ones((row_Y, n_segments))
    for j in range(0,self.n_products):
        for c in range(0,self.n_categories[j]):
            #get for every product, every category the independent probability p_js
            prob_j_c = np.power(p_js[:, j, c][np.newaxis, :], Y[:, j][:, np.newaxis] == c)
            #for each product and category, multiplicate p_js by the 'total' probabilty
            P_y_given_s = np.multiply(P_y_given_s, prob_j_c)

    return P_y_given_s

def prob_P_s_given_Z(self, param, shapes, Z, n_segments):   
    """
    Parameters
    ----------
    param : 1D array
            parameters of the HMM
    shapes : 2D array
            array consisting of shape and size of every single parameter matrix  
    Z : 2D array
        covariates of the HMM
    n_segments : int
            number of segments being used for the estimation of the HMM
    Returns
    -------
    P_s_given_Z : 2D array
         probabilities P(S_{i0} = s | Z_{it}), rows representing customers and columns representing columns

    """
    """function to compute probabilties P(X_{i0} = s| Z_{i0}) with parameters gamma_0"""
    
    #get number of customers for which P(S_{i0} = s | Z_{it}) is calculated
    row_Z = len(Z)
    
    #get the parameters in matrices instead of one vector
    gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)
    
    #get the logodds by adding the constant to the multiplication of the parameters and covariates
    P_s_given_Z = gamma_0[:,0][:,np.newaxis] + gamma_0[:,1:].dot(np.transpose(Z)) 
    #add the base case with zeros
    P_s_given_Z = np.vstack( (P_s_given_Z, np.zeros((1,row_Z))) )  
        
    #transform the logodds values to real probabilties P(S_{i0} = s | Z_{it}) 
    P_s_given_Z = np.transpose( np.exp( P_s_given_Z - logsumexp(P_s_given_Z) ))   
    
    return P_s_given_Z
        
    
def prob_P_s_given_r(self, param, shapes, Z, n_segments):
    """
    Parameters
    ----------
    param : 1D array
            parameters of the HMM
    shapes : 2D array
            array consisting of shape and size of every single parameter matrix  
    Z : 2D array
        covariates of the HMM
    n_segments : int
            number of segments being used for the estimation of the HMM
    Returns
    -------
    P_s_given_r : 3D array
         probabilities P(S_{it} = s | S_{it-1} = r), with first index representing customers, second representing new segment, third representing origin segment

    """
    """function to compute probabilties P(S_{it} = s | S_{it-1} = r) with parameters gamma_sr_0 and gamma_sk_t"""
    
    #get number of customers for which P(S_{it} = s | S_{it-1} = r)  is calculated
    row_Z = len(Z)

    #get the parameters in matrices instead of one vector
    gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)

    #add the base case to the transition parameters without covariates
    gamma_sr_0 = np.vstack((gamma_sr_0, np.zeros((1,n_segments))))
    
    #make for every customers a matrix (they are combined in one 2D matrix)
    P_s_given_r = np.repeat(gamma_sr_0[np.newaxis,:,:], row_Z, axis = 0)
        
    #get for every customer specific the values of multiplicating the covariates to the parameter gamma_sk_t in a matrix
    mat = np.matmul(gamma_sk_t, np.transpose(Z)) 
    mat = np.vstack((mat, np.zeros((1,row_Z))))
    mat = np.repeat(mat, n_segments, axis = 1)
    mat = np.array_split(mat, row_Z, axis = 1)
    mat = np.array(mat)
        
    #add the values of multiplicating the covariates to the parameter gamma_sk_t to the transition parameters without covariates
    P_s_given_r = P_s_given_r + mat

    #transform the logodds values to real probabilties P(S_{it} = s | S_{it-1} = r
    log_sum_exp = logsumexp(P_s_given_r, axis = 1, reshape = True)
    P_s_given_r = np.exp(P_s_given_r - log_sum_exp[:,np.newaxis,:]) #i x s x s

    return P_s_given_r
        


def gamma_sr_0_to_trans(self, param, shapes, n_segments):
    """
    Parameters
    ----------
    param : 1D array
            parameters of the HMM
    shapes : 2D array
            array consisting of shape and size of every single parameter matrix  
    n_segments : int
            number of segments being used for the estimation of the HMM
    Returns
    -------
    P_s_given_r : 3D array
         probabilities P(S_{it} = s | S_{it-1} = r), with first index representing new segment, second representing origin segment

    """
    """function to compute probabilties P(S_{it} = s | S_{it-1} = r) with parameters gamma_sr_0, so the 'base' transition probabilites (without using covariates)"""
    #get the parameters in matrices instead of one vector
    gamma_0, gamma_sr_0, gamma_sk_t, beta = param_list_to_matrices(self, n_segments, param, shapes)
    
    #add the base case to the transition parameters without covariates
    gamma_sr_0 = np.vstack((gamma_sr_0, np.zeros((1,n_segments))))
        
    #transform the logodds values to real probabilties P(S_{it} = s | S_{it-1} = r)
    P_s_given_r = np.exp(gamma_sr_0 - logsumexp(gamma_sr_0))
        
    return P_s_given_r
            

# =============================================================================
# --------------------function for maximisation step---------------------
# =============================================================================


def joint_event(self, alpha, beta, t, n_segments,
                P_s_given_Y_Z_ut, P_s_given_r, P_y_given_s):
    """
    Parameters
    ----------
    alpha : 3D array
        estimated probabilities P(Y_{0:t}, X_{t})
    beta  : 3D array
        estimated probabilities P(Y_{t+1:T} | X_{t})
    t : int
        time for which one wants to compute the return
    n_segments : int
        number of segments being used for the estimation of the HMM
    P_s_given_Y_Z_ut : 3D array
        element-wise multiplication of alpha and beta
    P_s_given_r : 3D array
        probabilities P(S_{it} = s | S_{it-1} = r), with first index representing new segment, second representing origin segment
    P_y_given_s : 2D array
        probabilities P(Y_{it} | X_{it} = s), with rows representing customers and columns representing segments 
    Returns
    -------
    P_sr_given_Y_Z : 3D array
         joint event probabilities P(X_{i,t-1} = s_{t-1}, X_{it} = s_t | Y_{it}, Z_{it}), with first index representing customers, second representing s_t, third representing s_{t-1}

    """
    """function to compute joint event P(X_{it-1} = s_{t-1}, X_{it} = s_t | Y_{it}, Z_{it})"""
      
    #initialise matrix
    P_sr_given_Y_Z = np.zeros((self.n_customers, n_segments, n_segments))
    
    #calculate for every customer for every origin, the probabilty transition to every new state s
    for s in range(0,n_segments):
        mat = np.multiply(np.transpose(alpha[:,:,t-1]), P_s_given_r[:,s,:])
        mat = np.multiply(mat, np.transpose([P_y_given_s[:,s]])) 
        mat = np.multiply(mat, np.transpose([beta[s,:,t]]))
        P_sr_given_Y_Z[:,s,:] = mat
    
    #transform the logodds values to real probabilties P(X_{it-1} = s_{t-1}, X_{it} = s_t | Y_{it}, Z_{it})
    sum_per_cust = np.sum(np.sum(P_sr_given_Y_Z, axis = 2), axis = 1)
    P_sr_given_Y_Z = np.divide(P_sr_given_Y_Z, sum_per_cust[:,np.newaxis, np.newaxis])
      
    return P_sr_given_Y_Z 


def state_event(self, alpha, beta):
    """
    Parameters
    ----------
    alpha : 3D array
        estimated probabilities P[Y_{0:t}, X_{t}]
    beta  : 3D array
        estimated probabilities P[Y_{t+1:T} | X_{t}]
    Returns
    -------
    P_s_given_Y_Z : 3D array
        probabilities P(X_{it} = s|Y_{it}, Z_{it}), with first index representing segments, second representing customers and third representing time
    """
    """function to compute P(X_{it} = s|Y_{it}, Z_{it})"""
    #get logodds of P_s_given_Y_Z by multiplicating alpha and beta
    P_s_given_Y_Z = np.multiply(alpha, beta) 
    
    #transform the logodds values to real probabilties P(X_{it} = s|Y_{it}, Z_{it})
    P_Y_given_Z = np.sum(P_s_given_Y_Z, axis = 0) 
    P_s_given_Y_Z = np.divide(P_s_given_Y_Z, P_Y_given_Z)
    
    return P_s_given_Y_Z


def logsumexp(x, axis = 0, reshape = False):
    """
    Parameters
    ----------
    x : array 
        array of which a sum needs to taken
    axis : int
        axis of which the sum needs to be taken
    reshape : boolean
        boolean that indicates whether a reshape is necessary to correctly take the sum
    Returns
    -------
    logsumexp : array
        array consisting of the sums of x
    """
    """function for avoiding the overflow/underflow when taking exponents (logs) of high (low) variables"""
    
    #get maximum of a predefined axis
    c = x.max(axis = axis)
    
    #get negative difference between value and maximum. Reshape if maximum has to reshaped (for instance if x is 3D array) 
    if reshape == False:
        diff = x - c 
    else:
        diff = x - c[:,np.newaxis,:]
        
    #get log of the sum of exponents of differences + the maximum
    logsumexp =  c + np.log(np.sum(np.exp(diff), axis = axis))
    return logsumexp









