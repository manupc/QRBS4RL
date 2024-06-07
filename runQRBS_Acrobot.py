#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:37:35 2024

@author: manupc
"""

import random
import numpy as np
from BinaryCHC import BinaryCHC as OptAlgorithm
from QRBS_RL_Evaluator import QRBS_RL_Evaluator as Evaluator
from Experimenter import Experimenter
from Wrappers import Acrobot_builder as env_builder


    
DEBUG= False

num_exper= 30
dataName= 'Acrobot-v1'
method= 'QRBS'
algorithmName= 'CHC'



nTests= 10 # Environment tests
RewardSolved= -80 # Reward for which the environment is considered solved
nPop= 50 # CHC's population size
nRules= 6 # CHC's solution number of rules
M= 5 # CHC's pseudo number of elitist best solution changes
r= 0.5 # CHC's percentaje of best solution change after divergence
MaxIterations= 1000 # CHC's Maximum number of iterations
MaxEvaluations= None # CHC's Maximum number of solutions evaluated

def ActionSelector(logits):
    
    aux= np.argmax(logits, axis=-1).reshape(-1)
    return aux

# Create evaluator and calculate solution size
evaluator= Evaluator(env_builder= env_builder, nTests= nTests, action_selector=ActionSelector,
                     nRules=nRules, reward_solved= RewardSolved)
solSize= evaluator.get_solution_size(nRules= nRules)


params= {}
params['n_pop']= nPop # -> Population size
params['sol_size']= solSize # -> Solution size
params['evaluator']= evaluator # -> Solution evaluator
params['M']= M # -> Number of elite solutions for divergence
params['r']= r #  -> Percentage in (0,1) of elite solutions mutation
params['maximization_problem']= True # -> True if the problem is maximization; False otherwise


execution= {}
execution['MaxIterations']= MaxIterations
execution['MaxEvaluations']= None
execution['verbose']= True


store_results= ['iterations', 'evaluations', 'best', 'best_fitness',
                'time', 'history_mean_fitness', 'history_best_fitness', 
                'nRules']
seed_initializer= lambda x: (np.random.seed(x), random.seed(x))

exprunner_param= {}
exprunner_param['algorithm']= OptAlgorithm                   # Algorithm's Python class with 'run' method
exprunner_param['alg_params']= params              # Dictionary containing algorithm construction method
exprunner_param['run_params']= execution           # Dictionary containing parameters for the run method
exprunner_param['algorithm_name']= algorithmName        #: Name of the algorithm
exprunner_param['problem_name']= dataName        #: Name of the problem to be solved
exprunner_param['additional_info']= method             #: Text containing additional info for the name of the output file
exprunner_param['runs']= 30                 #: Number of runs
exprunner_param['store_results']= store_results    #: List containing names of results to be saved
exprunner_param['output_file']= None               #: Output file for results (no extension). Default None
exprunner_param['seed_initializer']= seed_initializer #: Callable with input parameter the seed


runner= Experimenter(exprunner_param)


def calculate_nrules(results):
    best= results['best']
    nRules= evaluator.get_sol_nrules(best)
    results['nRules']= nRules
    return results

runner.run(force_restart=False, post_processing_outputs=calculate_nrules, debug=DEBUG)


