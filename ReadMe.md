# Predicting cross-sell opportunities for banking portfolios

Maintaining the relationship with existing customers and identifying opportunities for cross-selling is a key challenge for many businesses. In our paper, we present a Latent Markov approach to predict which customers should be targeted for acquiring certain financial portfolios. We apply this method to general portfolio data and portfolio transaction data provided by Dutch online bank Knab. Two separate models are used; one to model activity of customers and one to model product ownership.The model gives insight into which customer segments exist and what the probabilities to change between segments are based on the customer characteristics. We achieve high accuracy for predicting which portfolios customers will acquire next. Furthermore, a model to predict the extra cash inflow resulting from cross-selling is presented which can be used to make a targeting decision under a
constraint on the inflow of deposits.

## Instructions: 
Everything can be run from main.py. In order to perform specific tasks, make sure that the relevant variables at the beginning of the page are set to 'True'.


## Code file descriptions:
`<HMM_eff.py>` contains code to execute a Baum-Welch/forward-backward algorithm to estimate a Hidden Markov Model 
(with modelling the transition and initialisation probabilities by covariates) 
