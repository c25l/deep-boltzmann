deep_boltzmann
==============
This class implements a deep boltzmann machine as described by [Hinton and Salakhutdinov][1]. There could (might (should)) be a better way to get predictions, but a prediction method using gibbs sampling and masks is implemented.


It also implements a dropout back-propagation for some reasonable definition thereof using stochastic gradient descent.

The whole thing, is, of course, just a prototype, with no guarantees of anything, but it seems to work. The design is based heavily on use of numpy.einsum.


[1]: https://www.cs.toronto.edu/~hinton/absps/efficientDBM.pdf
