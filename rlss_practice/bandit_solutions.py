#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 20:11:07 2023

@author: matteo
"""
import numpy as np
from rlss_practice.model import Agent
from numpy.linalg import pinv

MEANS = np.array([0.1, 0.5, 0.9])
EPSILON = 0.1
_theta = np.array([0.45, 0.5, 0.5])
THETA = _theta / np.linalg.norm(_theta)
ALPHA = 2.

class EpsilonGreedy(Agent):
  def __init__(self, K, eps=EPSILON):
    self.eps = eps # exploration probability
    self.K = K # number of arms
    self.reset()

    self.cumulative_reward = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.avg_rewards = np.zeros(self.K)


  def reset(self):
    self.t = 0
    self.avg_rewards = np.zeros(self.K)
    self.cumulative_reward = np.zeros(self.K)
    self.num_played = np.zeros(self.K)

  def get_action(self):
    #SOLUTION
    u = np.random.random()
    if u<self.eps:
        return np.random.choice(self.K)
    #END SOLUTION

    chosen_arm_index = np.argmax(self.avg_rewards)
    return chosen_arm_index

  def receive_reward(self, chosen_arm, reward):
    self.cumulative_reward[chosen_arm] += reward
    self.num_played[chosen_arm] += 1
    self.avg_rewards[chosen_arm] = self.cumulative_reward[chosen_arm]/self.num_played[chosen_arm] # update

    self.t += 1

  def name(self):
    return 'EGreedy('+str(self.eps)+')'


class UCB(Agent):
  def __init__(self, K, alpha):
    self.alpha = alpha
    self.K = K
    self.reset()

    self.cumulative_reward = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.avg_rewards = np.zeros(self.K)


  def reset(self):
    self.t = 0
    self.avg_rewards = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.cumulative_reward = np.zeros(self.K)


  def get_action(self):
    #SOLUTION
    num_played = self.num_played + (1e-15 if np.any(self.num_played==0) else 0)
    bonuses = np.sqrt(self.alpha * np.log((self.t + 1)) / num_played)
    scores = self.avg_rewards + bonuses
    ##END SOLUTION

    chosen_arm_index = np.argmax(scores)
    return chosen_arm_index

  def receive_reward(self, chosen_arm, reward):
    self.cumulative_reward[chosen_arm] += reward
    self.num_played[chosen_arm] += 1
    self.avg_rewards[chosen_arm] = self.cumulative_reward[chosen_arm]/self.num_played[chosen_arm]

    self.t += 1

  def name(self):
    return 'UCB('+str(self.alpha)+')'

class ThompsonSampling(Agent):
  def __init__(self, K, sigma=1.):
    self.sigma = sigma #prior std
    self.K = K
    self.reset()

    self.cumulative_reward = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.avg_rewards = np.zeros(self.K)
    self.stds = self.sigma * np.ones(self.K) #TO FILL


  def reset(self):
    self.t = 0
    self.avg_rewards = np.zeros(self.K)
    self.cumulative_reward = np.zeros(self.K)
    self.num_played = np.zeros(self.K)
    self.stds = self.sigma * np.ones(self.K) #TO FILL

  def get_action(self):
    #SOLUTION:
    scores = np.random.normal(self.avg_rewards, self.stds)
    ##END SOLUTION
    chosen_arm_index = np.argmax(scores)
    return chosen_arm_index

  def receive_reward(self, chosen_arm, reward):
    self.cumulative_reward[chosen_arm] += reward
    self.num_played[chosen_arm] += 1
    self.avg_rewards[chosen_arm] = self.cumulative_reward[chosen_arm]/self.num_played[chosen_arm]

    self.stds[chosen_arm] = self.sigma / np.sqrt(self.num_played[chosen_arm]) #NOT GIVEN
    self.t += 1

  def name(self):
    return 'TS('+str(self.sigma)+')'


class LinEpsilonGreedy(Agent):
  def __init__(self, d,lambda_reg=1., eps=0.1,):
    self.eps = eps # exploration probability
    self.d = d
    self.lambda_reg = lambda_reg
    self.reset()

  def reset(self):
    self.t = 0
    self.hat_theta = np.zeros(self.d)

    #The covariance matrix is initialized here
    self.cov = self.lambda_reg * np.identity(self.d)

    #The inverse of the covariance matrix is initialized here
    self.invcov = np.identity(self.d)

    #The target vector is initialized here
    self.b_t = np.zeros(self.d)


  def get_action(self, arms):
    K, _ = arms.shape

    #Your code here
    u = np.random.random()
    if u<self.eps:
        return arms[np.random.choice(K)]

    estimated_means = np.dot(arms, self.hat_theta)
    chosen_arm_index = np.argmax(estimated_means)
    #end your code

    return arms[chosen_arm_index,:]

  def receive_reward(self, chosen_arm, reward):
    """
    update the internal quantities required to estimate the parameter theta using least squares
    """

    #Update inverse covariance matrix
    #your code:
    xxt = np.outer(chosen_arm, chosen_arm.T)
    self.cov += xxt
    self.invcov = pinv(self.cov)

    #Update the target vector

    self.b_t += reward * chosen_arm

    self.hat_theta = np.inner(self.invcov, self.b_t) # update the least square estimate

    #end your code
    self.t += 1

  def name(self):
    return 'LinEGreedy('+str(self.eps)+')'


class LinUCB(Agent):

    def __init__(self, d, delta, lambda_reg, alpha=1.):
        self.d = d
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.cov = self.lambda_reg * np.identity(d)


        self.alpha = alpha
        self.reset()

    def reset(self):
        # reset all local variables that should not be kept when the experiment is restarted
        self.t = 0
        self.hat_theta = np.zeros(self.d)
        self.cov = self.lambda_reg * np.identity(self.d)
        self.invcov = np.identity(self.d)
        self.b_t = np.zeros(self.d)


    def get_action(self, arms):
        K, _ = arms.shape
        self.UCBs = np.zeros(K)
       
        #IN THIS SOLUTION we have replaced the exploration parameter given in the notebook with a tighter one
        #self.beta = np.sqrt(self.lambda_reg) + np.sqrt(2*np.log(1./self.delta) + np.log(1+self.t/(self.d*self.lambda_reg)))
        
        #SOLUTION
        self.beta = np.sqrt(self.lambda_reg) + np.sqrt(2*np.log(1./self.delta) + np.log(np.linalg.det(self.cov) / self.lambda_reg**self.d))
       
        for i in range(K):
            a = arms[i,:]
            covxa = np.inner(self.invcov, a.T)
            self.UCBs[i] = np.dot(
                self.hat_theta,a) +  self.alpha * self.beta * np.sqrt(np.dot(a, covxa))
        #END SOLUTION
        chosen_arm_index = np.argmax(self.UCBs)

        
        chosen_arm = arms[chosen_arm_index]
        return chosen_arm


    def receive_reward(self, chosen_arm, reward):
       #SOLUTION
        xxt = np.outer(chosen_arm, chosen_arm.T)
        self.cov += xxt
        self.invcov = pinv(self.cov)

        self.b_t += reward * chosen_arm

        self.hat_theta = np.inner(self.invcov, self.b_t) # update the least square estimate

        #END OF SOLUTION
        self.t += 1

        pass


    def name(self):
        return "LinUCB("+str(self.alpha)+')'
    
    
class LinTS(Agent):

  def __init__(self, d, delta, lambda_prior):
    self.d = d
    self.delta = delta
    self.lambda_prior = lambda_prior
    self.cov = self.lambda_prior * np.identity(d)
    self.reset()

  def reset(self):
    # reset all local variables that should not be kept when the experiment is restarted
    self.t = 0
    self.hat_theta = np.zeros(self.d)
    self.cov = self.lambda_prior * np.identity(self.d)
    self.invcov = np.identity(self.d)
    self.b_t = np.zeros(self.d)


  def get_action(self, arms):
    """
        This function implements LinUCB
        Input:
        -------
        arms : list of arms (d-dimensional vectors)

        Output:
        -------
        chosen_arm : vector (chosen arm array of features)
        """

    K, _ = arms.shape
    estimated_means = np.zeros(K)

    hallucinated_theta = np.random.multivariate_normal(self.hat_theta, self.invcov)

    for i in range(K):
        estimated_means[i] = np.dot(hallucinated_theta,arms[i,:])


    #choose arm with largest estimated mean
    chosen_arm_index = np.argmax(estimated_means)
    chosen_arm = arms[chosen_arm_index]


    return chosen_arm



  def receive_reward(self, chosen_arm, reward):
    """
    update the internal quantities required to estimate the parameter theta using least squares
    """
    xxt = np.outer(chosen_arm, chosen_arm.T)
    self.cov += xxt
    self.invcov = pinv(self.cov)

    self.b_t += reward * chosen_arm

    self.hat_theta = np.inner(self.invcov, self.b_t) # update the least square estimate
    self.t += 1

    pass


  def name(self):
    return "LinTS"