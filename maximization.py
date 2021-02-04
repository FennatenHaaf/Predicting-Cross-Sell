# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:05:56 2021

@author: matth
"""
import numpy as np

#function for maximization step
def maximization_step_cov(self, alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in, T):
    """function for the maximization step"""
    
    """combine the separate parameter matrices to one vector"""
    n_gamma_0 = gamma_0_in.size
    n_gamma_sr_0 = gamma_sr_0_in.size
    n_gamma_sk_t = gamma_sk_t_in.size
    n_beta = beta_in.size
    
    sizes = np.array([n_gamma_0, n_gamma_sr_0, n_gamma_sk_t, n_beta])
    
    x0 = concatenate(gamma_0_in.flatten(), gamma__sr_0_in.flatten(), gamma_sk_t_in.flatten(), beta_in.flatten())
    
    """perform the maximization"""
    [param_out] = self.sci.optimize.minimize(optimization_function, x0, args=(alpha, beta, sizes, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in))
    
    """change the shape of the parameters back to separate matrices"""
    gamma_0_out = param_out[0:n_gamma_0]
    gamma_sr_0_out = param_out[n_gamma_0:(n_gamma_0+n_gamma_sr_0)]
    gamma_sk_t_out = param_out[(n_gamma_0+n_gamma_sr_0):(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t)]
    beta_out = param_out[(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t):x0.shape[0]]
    
    gamma_0_out = np.reshape(gamma_0_out, np.shape(gamma_0_in))
    gamma_sr_0_out = np.reshape(gamma_sr_0_out, np.shape(gamma_sr_0_in))                        
    gamma_sk_t_out = np.reshape(gamma_sk_t_out, np.shape(gamma_st_t_in))                        
    beta_out = np.reshape(beta_out, np.shape(beta_in))
    
    
    return gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out


def optimization_function(self, x, alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in, sizes):
    """function that has to be minimized"""
    
    n_segments = beta.shape[0]

    """change the shape of the parameters back to separate matrices"""
    n_gamma_0 = sizes[0]
    n_gamma_sr_0 = sizes[1]
    n_gamma_sk_t = sizes[2]
    n_beta = sizes[3]
    
    gamma_0 = x[0:n_gamma_0]
    gamma_sr_0 = x[n_gamma_0:(n_gamma_0+n_gamma_sr_0)]
    gamma_sk_t = x[(n_gamma_0+n_gamma_sr_0):(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t)]
    beta_ = x[(n_gamma_0+n_gamma_sr_0+n_gamma_sk_t):x0.shape[0]]
    
    gamma_0 = np.reshape(gamma_0, np.shape(gamma_0_in))
    gamma_sr_0 = np.reshape(gamma_sr_0_out, np.shape(gamma_sr_0_in))                        
    gamma_sk_t = np.reshape(gamma_sk_t_out, np.shape(gamma_st_t_in))                        
    beta_ = np.reshape(beta_out, np.shape(beta_in))
    
    """compute function"""
 




    sum = 0;
    for i in range(0,self.n_customers):   
        Y = self.list_Y[0][i,:]
        Z = self.list_Z[0][i,:]        
        
        P_s_given_Z = self.prob_P_s_given_Z(gamma_0, Z) 
        P_s_given_Y_Z = self.state_event(alpha,beta,i,0)
        
        sum = sum + np.sum(multiply(P_s_given_Y_Z, math.log(P_s_given_Z)))
        
        for t in range(1,self.T):
            Y = self.list_Y[t][i,:]
            Z = self.list_Z[t][i,:]  
            P_s_given_r = self.prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z)

            for r in range(0,n_segment):
                for s in range(0,n_segment):
                    
                    
                    P_sr_given_Y_Z = self.joint_event(y, z, alpha, beta, beta_in, gamma_sr_0_in, gamma_sk_t_in, i, t, s, r)
                    P_s_given_r = P_s_given_r[s,r]
                    
                    sum = sum + P_sr_given_Y_Z * math.log(P_s_given_r)

        for t in range(0,self.T):
            for s in range(0,n_segment):
                P_s_given_Y_Z = self.state_event(alpha,beta,i,t)
                P_y_given_s = self.prob_P_y_given_s(y, self.prob_pi(beta_))

                sum = sum + P_s_given_Y_Z*log(P_y_given_s)

    return sum    


def joint_event(self, y, z, alpha, beta, beta_in, gamma_sr_0_in, gamma_sk_t_in, i, t, s, r):
    """function to compute P(X_it-1 = s_t-1, X_it = s_t|Y_i, Z_i)"""
    
    n_segments = beta.shape[0]
    P_s_given_Y_Z = np.zeros((n_segments))
    P_s_given_Y_Z = np.multiply(alpha(:,i,t), beta(:,i,t))
    
    P_s_given_r = prob_P_s_given_r(gamma_sr_0, gamma_sk_t, Z)
    
    P_y_given_s = self.prob_P_y_given_s(y, self.prob_pi(beta_in))

    P_sr_given_Y_Z = ( alpha[r,i,t-1] * P_s_given_r[s,r] * P_y_given_s * beta[s,i,t] ) / np.sum(P_s_given_Y_Z)
    


def state_event(self, alpha, beta, i, t):
    """function to compute P(X_it = s|Y_i, Z_i)"""

    n_segments = beta.shape[0]

    P_s_given_Y_Z = np.zeros((n_segments))
    P_s_given_Y_Z = np.multiply(alpha(:,i,t), beta(:,i,t))
    P_s_given_Y_Z = P_s_given_Y_Z/np.sum(P_s_given_Y_Z)
    
    return P_s_given_Y_Z

