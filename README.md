# QRBS4RL
Quantum Rule-Based Systems to solve Reinforcement Learning problems

This repository contains the source code for the article "Automatic evolutionary design of Quantum Rule-Based Systems and applications to Quantum Reinforcement Learning". If you plan to use this code, please cite:


Cuellar, M.P., Pegalajar, M.C., Cano, C.: Automatic evolutionary design of quantum rule-based systems and applications to quantum reinforcement learning. Quantum Information Processing 23(5), 179 (2024) https://doi.org/10.1007/s11128-024-04391-0

- The file BinaryCHC.py contains a Python implementation of the binary CHC evolutionary algorithm.
- The file QRBS.py implements the Quantum Rule-Based System.
- The file Evaluator.py implements the fitness evaluation function of the QRBS for Quantum Reinforcement Learning using TensorFlow Quantum v0.7.0.
- The file Wrappers.py include the preprocessing wrappers explained in the article for each studied environment.
- The remaining files QRBS_*.py contain the main files executed for the experiments in the article.

