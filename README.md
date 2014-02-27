deep_boltzmann
==============
This class implements a deep boltzmann machine as described by [Hinton and Salakhutdinov][1]. There could (might (should)) be a better way to get predictions, but a prediction method using gibbs sampling and masks is implemented.


Planned updates include: 
1.  Greedy layerwise pretraining

2.  [AIS estimation of partition function][2]

3.  Use of DBM to initialize a traditional neural network.


It also implements a dropout back-propagation for some reasonable definition thereof using stochastic gradient descent.

The whole thing, is, of course, just a prototype, with no guarantees of anything, but it seems to work. The design is based heavily on use of numpy.einsum.


[1]: https://www.cs.toronto.edu/~hinton/absps/efficientDBM.pdf
[2]: http://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf
