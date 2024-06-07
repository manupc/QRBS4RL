#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:03:15 2024

@author: manupc
"""

import numpy as np
from QRBS import QRBS

class QRBS_RL_Evaluator:
    
    """
    env_builder: Callable to build an environment containing fields nInputs and nOutputs
    nTests: Number of tests to assess a solution performance
    action_selector: callable containing a batch of GQRS outputs to select an action
    nRules: Number of maximum number of rules allowed in the GQRS
    reward_solved: Avg Return to set if the problem is solved. None if not applicable
    """
    def __init__(self, env_builder, nTests, action_selector, nRules, reward_solved= None):
        
        self.__env_builder= env_builder
        self.__nTests= nTests
        self.__action_selector= action_selector
        self.__nRules= nRules
        self.__reward_solved= reward_solved
        self.__lastEvaluationFitness= None
        
        
        # Test envs
        self.__tsEnvs= [env_builder() for _ in range(nTests)]
        
        aux_env= env_builder()
        self.__nInQubits= aux_env.nInputs
        self.__nOutQubits= aux_env.nOutputs
        


    """
    Returns True if the problem is solved, False otherwise
    """
    def problem_solved(self):
        return self.__reward_solved is not None and self.__lastEvaluationFitness >= self.__reward_solved



    """
    Returns the number of decision variables required for the ruleset representation
    """
    def get_solution_size(self, nRules= None):
        return self.get_solution_binary_size(nRules)



    """
    Returns the number of binary decision variables required for the ruleset representation
    """
    def get_solution_binary_size(self, nRules= None):
        if nRules is None:
            nRules= self.__nRules
        tam_antecedent= 2*self.__nInQubits
        tam_consequent= int(np.ceil( np.log2(self.__nOutQubits) ))  # Binary representation of target qubit
            
        return (tam_antecedent + tam_consequent +1)*nRules # +1 to activate/deactivate the rule



    """
    Returns the number of active rules for a given solution
    """
    def get_sol_nrules(self, solution):
        return int(np.sum( solution.reshape(self.__nRules, -1)[:, 0]))


    """
    Returns a ruleset of a solution in a GQRS format
    """
    def getRuleSet(self, solution):
        
        rules= solution.reshape(self.__nRules, -1)
        ruleset= []
        last_pos_antecedent= 1 + self.__nInQubits*2
        
        for rule_sol in rules:
            if int(rule_sol[0]) == 0: # Checks if rule is active
                continue

            consequent_sol= rule_sol[last_pos_antecedent:]
            target_qubit= np.dot(consequent_sol, 1 << np.arange(len(consequent_sol))[::-1])
            
            
            if target_qubit >= self.__nOutQubits: # Invalid target: Rule not active
                continue

            
            # It is active
            consequent= [target_qubit]

            antecedent= []                
            for current_in_qubit, i in enumerate(range(1, last_pos_antecedent, 2)):
                if int(rule_sol[i]) == 1:
                    control= int(rule_sol[i+1])
                    antecedent.append( [current_in_qubit, control] )
            
            ruleset.append( [antecedent, consequent] )
        return ruleset



    """
    Returns a GQRS model for a given solution
    """
    def getModelForSolution(self, solution):
        ruleset= self.getRuleSet(solution)
        model= QRBS(nInputQubits= self.__nInQubits, nOutputQubits= self.__nOutQubits, 
                    rule_list= ruleset)
        return model
    
    
    
            
        
    
    """
    Evaluates a solution. Returns the fitness and number of evaluations 
    """
    def __call__(self, solution):
        
        model= self.getModelForSolution(solution)
        
        envs= self.__tsEnvs
        
        
        # Initialize test environments
        active_envs= list(range(self.__nTests))
        R_test= [0]*self.__nTests
        S= []
        for i, env in enumerate(envs):
            s, _= env.reset()
            S.append(s)
        
        # Run environments in pseudo-parallel
        while active_envs:
            
            # Get action
            inputs= [S[i] for i in active_envs]
            #t_inputs= tf.convert_to_tensor(inputs, dtype=tf.float32)

            logits= model(inputs)
            actions= self.__action_selector(logits.numpy())
            
            #actions= np.clip(actions, a_min=-1.0, a_max=1.0)
            remove_envs= []
            for action, env_idx in zip(actions, active_envs):
                
                sp, reward, terminated, truncated, _ = envs[env_idx].step(action)
                R_test[env_idx]+= reward
                
                done= terminated or truncated
                if done:
                    remove_envs.append(env_idx)
                else:
                    S[env_idx]= sp
            for env_idx in remove_envs:
                active_envs.remove(env_idx)
                
        fitness= np.mean(R_test)
        self.__lastEvaluationFitness= fitness
        
        return fitness, 1
    