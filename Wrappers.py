#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:54:36 2024

@author: manupc
"""
import gymnasium as gym
import numpy as np



##################################################################
### BLACK JACK
##################################################################

"""
Observation Wrapper for BlackJack env
"""
class BlackJackObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.tupleBits= []
        self.nBits= 0
        for t in self.observation_space:

            nbits= int( np.ceil( np.log2( t.n ) ) )
            self.tupleBits.append( nbits )
            self.nBits+= nbits
        self.observation_space= gym.spaces.Box(low=0, high=1, shape=(self.nBits,), dtype=np.float32)
        self.nInputs= self.observation_space.shape[0]
        self.nOutputs= env.action_space.n

    
    
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        new_obs= np.zeros(self.nBits, dtype= np.float32)
        
        cur_o= obs[0]
        b= reversed(bin(cur_o)[2:])
        for i, v in enumerate(b):
            new_obs[self.nBits-1-i]= int(v)
            
        cur_o= obs[1]
        b= reversed(bin(cur_o)[2:])
        for i, v in enumerate(b):
            new_obs[self.nBits-1-i-self.tupleBits[0]]= int(v)
        
        cur_o= obs[2]
        b= reversed(bin(cur_o)[2:])
        for i, v in enumerate(b):
            new_obs[self.nBits-1-i-self.tupleBits[0]-self.tupleBits[1]]= int(v)

        return new_obs


BlackJack_builder= lambda render=False:BlackJackObervationWrapper(gym.make('Blackjack-v1', render_mode='rgb_array' if render else None))




##################################################################
### CARTPOLE
##################################################################
"""
Observation Wrapper for CartPole env
"""
class CartPoleObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.nInputs= env.observation_space.shape[0]
        self.nOutputs= env.action_space.n

   
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        new_obs= np.empty(len(obs), dtype=np.float32)
        new_obs[0]= (obs[0]/4.8+1)/2
        new_obs[1]= (2*np.arctan(obs[1])/np.pi + 1)/2
        new_obs[2]= ((obs[2]/0.418)+1)/2
        new_obs[3]= (2*np.arctan(obs[3])/np.pi + 1)/2
        return new_obs

CartPole_builder= lambda render=False: CartPoleObervationWrapper(gym.make('CartPole-v1', render_mode='rgb_array' if render else None))




##################################################################
### ACROBOT
##################################################################

"""
Observation Wrapper for Acrobot env
"""
class AcrobotObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.nInputs= env.observation_space.shape[0]
        self.nOutputs= env.action_space.n

   
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        new_obs= np.empty(len(obs), dtype=np.float32)
        new_obs[0]= (obs[0]+1)/2
        new_obs[1]= (obs[1]+1)/2
        new_obs[2]= (obs[2]+1)/2
        new_obs[3]= (obs[3]+1)/2
        new_obs[4]= (obs[4]/(4*np.pi)+1)/2
        new_obs[5]= (obs[5]/(9*np.pi)+1)/2
        
        return new_obs

Acrobot_builder= lambda render=False: AcrobotObervationWrapper(gym.make('Acrobot-v1', render_mode='rgb_array' if render else None))




##################################################################
### MOUNTAIN CAR
##################################################################
"""
Observation Wrapper for MountainCar env
"""
class MountainCarObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.nInputs= env.observation_space.shape[0]
        self.nOutputs= env.action_space.n

   
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):

        new_obs= np.empty(self.nInputs, dtype=np.float32)
        new_obs[0]= (obs[0]+1.2)/1.8
        new_obs[1]= (obs[1]+0.07)/0.14
        return new_obs


MountainCar_builder= lambda render=False: MountainCarObervationWrapper(gym.make('MountainCar-v0', render_mode='rgb_array' if render else None))




##################################################################
### FROZEN LAKE
##################################################################

"""
Observation Wrapper for FrozenLake env
"""
class FrozenLakeObervationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, nrows, ncols):
        super().__init__(env)
        self.nrows= nrows
        self.ncols= ncols
        self.nBitsRows= int(np.ceil(np.log2( nrows )))
        self.nBitsCols= int(np.ceil(np.log2( ncols )))
        self.nBits= self.nBitsCols + self.nBitsRows
        self.observation_space= gym.spaces.Box(low=0, high=1, shape=(self.nBits,), dtype=np.float32)
        self.nInputs= self.observation_space.shape[0]
        self.nOutputs= env.action_space.n
        
    
    
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        new_obs= np.zeros(self.nBits, dtype= np.float32)
        b= reversed(bin(obs % self.ncols)[2:])
        for i, v in enumerate(b):
            new_obs[self.nBits-1-i]= int(v)
        b= reversed(bin(obs // self.ncols)[2:])
        for i, v in enumerate(b):
            new_obs[self.nBits-1-i-self.nBitsCols]= int(v)
        return new_obs

FrozenLakeNonSlippery_builder= lambda render=False: FrozenLakeObervationWrapper(gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array' if render else None), 
                                                         nrows=4, ncols=4)

FrozenLakeSlippery_builder= lambda render=False: FrozenLakeObervationWrapper(gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array' if render else None), 
                                                         nrows=4, ncols=4)

