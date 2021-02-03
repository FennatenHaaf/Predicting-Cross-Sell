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
    gamma_0_out, gamma_sr_0_out, gamma_sk_t_out, beta_out = maximization_step(Y, Z, , alpha, beta, gamma_0_in, gamma_sr_0_in, gamma_sk_t_in, beta_in, T)
    initialstateprob = ones(N,S)
    transitionmatrix = ones(N,S)
    stateprob = ones(NS,S)
    for i in range(1,N):
        stateprob[i] = joint_event(beta_out)
        [initialstateprob[i], transitionmatrix[i]] = state_event(Z, gamma_0_out, gamma_sr_0_out)
    return 
    
def joint_event(beta_out):
    pi = ones(J, S)
    for j in range(1,J):
        for s in range(1,S):
            pi[j,s] = 1/(1+exp(sum(beta_out[])))
    return pi

def state_event(Z, gamma_0_out, gamma_sr_0_out):
    gamma_0_val = gamma_0_out*Z
    S_0 = exp(gamma_val[S])
    initial = ones((S,1))
    for i in range(1, S-1):
        initial[i] = exp(gamma_val[i])*S
    gamma_val = gamma_sr_0_out*Z
    prob = ones(())
    for t in range(1,T):
        s_t[t] = exp(gamma_val[t])
        prob[] = exp(gamma_val[t])*s_t[t]
    return prob, initial
