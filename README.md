NEAT Library
===========
Author: James Easton
----------
This project is an implementation of the NEAT algorithm in Python. It is meant to act as a library so that the algorithm can be implemented in other projects easily. This project was created by me so that I could become more familiar with the NEAT algorithm and to practive reading research papers and implementing them in practice. This library is incomplete and is not in running form yet, but it should be finished soon. 

Files
----------
NEAT.py - This file handles everything specific to the NEAT algorithm, from performing crossover between two neural networks to calculating the excess and disjoint of two neural nets.
NeuralNet.py - This file contains everything necessary to create a traditional neural network object.
Node.py - This file contains the node object used in neural nets.
Connection.py - This file contains the connection object used by neural nets to link nodes. Aside from being a standard graph edge, it also has an innovation number which is needed by the NEAT algorithm.