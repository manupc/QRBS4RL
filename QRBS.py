#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:59:41 2024

@author: manupc
"""

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
import sympy


"""
Generalized Quantum Rule System
"""
class QRBS(tf.keras.Model):
    
    """
        nInputQubits: Number of input qubits
        nOutputQubits: Number of output qubits
        allow_input_rotations: True to allow rotation on input qubits
        allow_output_rotations: True to allow rotation on output qubits
        default_output_value: Scalar in [0,1] to set the default output rotation for each output qubit. None if not applicable
        rule_list: List of rules. Each rule is in the following format: [ [antecedent], [consequent]], where
            - antecedent is a sequence of nInputQubits pairs
                [number, rotation] if allow_input_rotations is True
                [number, control] if allow_input_rotations is False
                    number: index of an input qubit to be used as control
                    rotation: Degree of rotation in [0, 1]
                    control: 0/1 to set the control state
            - consequent is a tuple:
                [] if nOutputQubits=1 and allow_output_rotations=False
                [rotation] if nOutputQubits=1 and allow_output_rotations=True
                [output]  if nOutputQubits>1 and allow_output_rotations=False
                [output, rotation]  if nOutputQubits>1 and allow_output_rotations=True
                where:
                    output is an index of a target output qubit in 0..nOutputQubits-1
                    rotation: Rotation in [0,1]
            
        allow_input_rotations: True to activate Rx rotations in inputs
        allow_output_rotations: True to activate CRX output rotations 
    """
    def __init__(self, nInputQubits : int, nOutputQubits : int, rule_list : list):
        super().__init__()
        self.__nInQubits= nInputQubits
        self.__nOutQubits= nOutputQubits
        self.__nQubits= self.__nInQubits + self.__nOutQubits
        
        self.__rules= rule_list
        
        # Create the circuit and get the circuit object and the name of input and parameter circuit data
        self.__qubits = cirq.GridQubit.rect(1, self.__nQubits)
        self.__circuit, self.__inSym= self.create_circuit()
        
        # Create the observable to be measured
        self.__observables= [cirq.Z(self.__qubits[q]) for q in range(self.__nInQubits, self.__nQubits)]
        
       # Sort symbols to set W and Wi properly
        symbols = [str(symb) for symb in self.__inSym]
        self.__indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        # Create the tensorflow computation layer for the circuit
        self.__empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.__computation_layer = tfq.layers.ControlledPQC(self.__circuit, self.__observables)
        self.__tiled_up_circuits= None

        self.__batch_dim= 0
        self.__joined_vars= None


    def print_circuit(self):
        print('\n\nCircuit:\n')
        print(self.__circuit)
        print()



    # Execution of thw model for the given inputs
    def call(self, inputs):
        
        # get the number of input patterns, and prepare the batch for parallel execution
        batch_dim = tf.gather(tf.shape(inputs), 0)
        if batch_dim != self.__batch_dim:
            self.__tiled_up_circuits = tf.repeat(self.__empty_circuit, repeats=batch_dim)
            self.__batch_dim= batch_dim
        
        joined_vars = tf.gather(inputs, self.__indices, axis=1)
        
        
        out= self.__computation_layer([self.__tiled_up_circuits, joined_vars])
        return out


    # Creates the circuit of GQRS model 
    def create_circuit(self):
        
        circuit_description= self.__rules
        nOutQubits= self.__nOutQubits
        nInQubits= self.__nInQubits
        qubits= self.__qubits
        
        # Auxiliary
        unused_qubits= [True]*nOutQubits
        
        # Inputs
        inSym= sympy.symbols(f'x(0:{nInQubits})')
        inSym= np.asarray(inSym).reshape(nInQubits)
        
        # Input embedding Circuit 
        circuit = cirq.Circuit()
        circuit += cirq.Circuit(cirq.rx(np.pi*inSym[i])(qubits[i]) for i in range(nInQubits))


        # Rules
        for layer_idx in range(len(circuit_description)):
            
            layer= circuit_description[layer_idx]

            # get Consequent
            consequent= np.array(layer[1], copy=False)
            target_qubit= 0 if nOutQubits==1 else int(consequent[0])
            
            if target_qubit >= nOutQubits: # Invalid target
                continue

            unused_qubits[target_qubit]= False
            target_qubit+= nInQubits

            
            # Antecedent rotations
            antecedent= layer[0]
            
            controls= []
            ctrl_states= []
            for element in antecedent:
                controls.append( int(element[0]) )
                ctrl_states.append( int(element[1])) 
            
            
            # Create controlled gate rule
            rule_qubits= [qubits[c] for c in controls]
            rule_qubits.append(qubits[target_qubit])
            
            CGate = cirq.ControlledGate(sub_gate=cirq.X, num_controls=len(controls), control_values= ctrl_states)
            circuit+= CGate(*rule_qubits)

        # Set I operation on unused outputs
        for i in range(nOutQubits):
            if unused_qubits[i]:
                circuit+= cirq.I(qubits[nInQubits+i])
        
        return circuit, list(inSym.flat)

