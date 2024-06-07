#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:02:14 2024

@author: manupc
"""

import numpy as np
import time


"""
Implementation of the Binary CHC algorithm
"""
class BinaryCHC:
    
    """
    params: Dictionary containing:
        params['n_pop'] -> Population size
        params['sol_size'] -> Solution size
        params['evaluator'] -> Solution evaluator
        params['M'] -> Number of elite solutions for divergence
        params['r'] -> Percentage in (0,1) of elite solutions mutation
        params['maximization_problem'] -> True if the problem is maximization; False otherwise
    """
    def __init__(self, params):
        
        n_pop= params['n_pop']
        sol_size= params['sol_size']
        sol_evaluator= params['evaluator']
        M= params['M']
        r= params['r']
        maximization_problem= params['maximization_problem']
        
        
        assert(n_pop> 0 and n_pop%2 == 0 and sol_size > 0)
        assert(1<=M<n_pop)
        assert(0<r<=1)
        
        self.__maximization_problem= maximization_problem
        self.__evaluator= sol_evaluator
        self.__n_pop= n_pop
        self.__sol_size= sol_size
        self.__M= M
        self.__r= r

        self.__stopping_criterion= False
        self.__MaxIterations= None
        self.__MaxEvaluations= None
        self.__problem_solved= False
        self.__betterThan= self.__greaterThan if self.__maximization_problem else self.__lessThan
        
    
    
    def __lessThan(self, f1, f2):
        return f1 < f2
    


    def __greaterThan(self, f1, f2):
        return f1 > f2
    
        
    
    def __initialize_pop(self):
        self.__recombination_threshold= self.__sol_size/4
        
        pop= np.random.randint(low= 0, high= 2, size=(self.__n_pop, self.__sol_size), dtype= np.uint8)
        fitness= np.empty(shape=(self.__n_pop), dtype= np.float32)
        fitness[:]= np.nan
        self.__best= np.empty(shape=(self.__sol_size), dtype= np.uint8)
        self.__best_fitness= None
        self.__problem_solved= False
        return pop, fitness
        
        
    
    def __update_best(self, solution, fitness):
        copy= self.__best_fitness is None or \
              (self.__maximization_problem and fitness > self.__best_fitness) or \
              (not self.__maximization_problem and fitness < self.__best_fitness)
        if copy:
            
            self.__best[:]= solution[:]
            self.__best_fitness= fitness
            current_time= time.time()-self.__t0
            history_record= ( self.__it, self.__evaluations, current_time, self.__best_fitness)
            self.__history_best.append( history_record )
        

        
    def __HUX(self, p1, p2):

        c1= p1.copy()
        c2= p2.copy()
        differents= np.where(p1!= p2)[0]
        half= len(differents)//2
        if half == 0:
            half= len(differents)
        changes= np.random.permutation(differents)[:half]
        c1[changes]= p2[changes]
        c2[changes]= p1[changes]

        return c1, c2
        

        
    def __diverge(self):
        
        self.__recombination_threshold= self.__r*(1-self.__r)*self.__sol_size
        pop= np.empty(shape=(self.__n_pop, self.__sol_size), dtype=np.uint8)
        fitness= np.empty(shape=(self.__n_pop), dtype= np.float32)
        pop[:self.__M, :]= self.__best[:]
        fitness[0]= self.__best_fitness
        flips= int(np.round(self.__r*self.__sol_size))

        for i in range(1, self.__M):
            perm= np.random.permutation(self.__sol_size)[:flips]
            pop[i, perm]= 1-pop[i, perm]
        
        pop[self.__M:]= np.random.randint(low= 0, high= 2, size=(self.__n_pop-self.__M, self.__sol_size), dtype= np.uint8)
        self.__evaluate(pop[1:], fitness[1:])
        return pop, fitness
        
        
    
    def __update_stopping_criterion(self):
        if (self.__MaxIterations is not None and self.__it >= self.__MaxIterations) or\
            (self.__MaxEvaluations is not None and self.__evaluations >= self.__MaxEvaluations) or\
            self.__problem_solved:
            self.__stopping_criterion= True
        return self.__stopping_criterion
    
    
    
    def __evaluate(self, pop, fitness):
    
        for i, sol in enumerate(pop):
            fitness[i], evals= self.__evaluator(sol)
            self.__evaluations+= evals
            if self.__evaluator.problem_solved():
                self.__problem_solved= True
            """
            if self.__update_stopping_criterion():
                break
            """
            
        # Population sort
        new_idxs= np.argsort(fitness)
        if self.__maximization_problem:
            new_idxs= new_idxs[::-1]

        fitness[:]= fitness[new_idxs]
        pop[:]= pop[new_idxs]
        
        # Update best sol
        self.__update_best(pop[0], fitness[0])
        


    def __merge_pop(self, pop, fitness, children, fitness_children):

        i, j= 0, 0
        new= False
        newpop= []
        newfit= []
        while i<len(pop) and j<len(children) and len(newpop)< self.__n_pop:
            if self.__betterThan(fitness[i], fitness_children[j]) or fitness[i] == fitness_children[j]:
                newpop.append(pop[i])
                newfit.append(fitness[i])
                i+= 1
            else:
                new= True
                newpop.append(children[j])
                newfit.append(fitness_children[j])
                j+= 1
                
        while i<len(pop) and len(newpop)< self.__n_pop:
            newpop.append(pop[i])
            newfit.append(fitness[i])
            i+= 1
            
        while j<len(children) and len(newpop)< self.__n_pop:
            newpop.append(children[j])
            newfit.append(fitness_children[j])
            j+= 1
            new= True
        
        newpop= np.array(newpop, copy=False, dtype=np.uint8)
        newfit= np.array(newfit, copy= False, dtype=np.float32)
        
        return newpop, newfit, new

    
    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['MaxEvaluations']: Approximated Maximum number of solution's evaluations
        params['verbose']: True to show results in Console
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['evaluations'] -> Number of solution evaluations
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iterations, evaluations, current_time, value) of mean fitness per iteration
            out['history_best_fitness'] -> History (iterations, evaluations, current_time, value) of best fitness update
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        MaxEvaluations= params['MaxEvaluations']
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        
        
        assert(MaxIterations is not None or MaxEvaluations is not None)
        self.__stopping_criterion= False
        self.__MaxIterations= MaxIterations
        self.__MaxEvaluations= MaxEvaluations
        N= self.__n_pop # Population size
        

                
        # Initialization
        self.__evaluations= 0
        self.__it= 0
        history_mean= []
        self.__history_best= []

        self.__t0= time.time()
        pop, fitness= self.__initialize_pop()
        self.__evaluate(pop, fitness)
        
        current_time= time.time()-self.__t0
        m= np.mean(fitness)
        history_record= ( self.__it, self.__evaluations, current_time, m)
        history_mean.append( history_record )
        if verbose:
            print(verbose_text_append+' BEGIN. It. {}, Eval {}. Mean {:.3f}. Best {:.3f}. t= {:.2f}'.format(self.__it, self.__evaluations, m, self.__best_fitness, current_time))
        
        # Main Loop
        while not self.__update_stopping_criterion():
            self.__it+= 1 # Next Iterator
            
            # Selection
            pair_parents_idx= np.random.permutation(N).reshape(-1, 2)
            
            # Recombination
            children= []
            for i, parents_idx in enumerate(pair_parents_idx):
                
                
                p1, p2= pop[parents_idx[0]], pop[parents_idx[1]]
                parents_can_recombine= np.sum(np.abs(p1-p2)) >= self.__recombination_threshold
                if parents_can_recombine:
                    c1, c2= self.__HUX(p1, p2)
                    children.extend([c1, c2])

            

            # Chech convergence
            if len(children) > 0:

                children= np.array(children, copy= False, dtype=np.uint8)
                fitness_children= np.empty(shape=(len(children)), dtype=np.float32)
                
                # Offsprint evaluation
                self.__evaluate(children, fitness_children)

                # Replacement
                pop, fitness, different_population= self.__merge_pop(pop, fitness, children, fitness_children)

            else:
                different_population= False
            
            # Reduce Threshold if convergence
            if not different_population:
                self.__recombination_threshold-= 1
                
            # Check diverge
            if self.__recombination_threshold < 0:
                pop, fitness= self.__diverge()
            
            current_time= time.time()-self.__t0
            m= np.mean(fitness)
            history_record= ( self.__it, self.__evaluations, current_time, m)
            history_mean.append( history_record )
            if verbose:
                print(verbose_text_append+' It. {}, Eval {}. Mean {:.3f}. Best {:.3f}. t= {:.2f}'.format(self.__it, self.__evaluations, m, self.__best_fitness, current_time))
        
        current_time= time.time()-self.__t0
        history_record= ( self.__it, self.__evaluations, current_time, m)
        history_mean.append( history_record )
        if verbose:
            print(verbose_text_append+' END. It. {}, Eval {}. Best {:.2f}. t= {:.2f}'.format(self.__it, self.__evaluations, self.__best_fitness, current_time))
            
        out= {}
        out['iterations']= self.__it
        out['evaluations']= self.__evaluations
        out['best']= self.__best
        out['best_fitness']= self.__best_fitness
        out['time']= current_time
        out['history_mean_fitness']= history_mean
        out['history_best_fitness']= self.__history_best
        self.__history_best= None

        return out
