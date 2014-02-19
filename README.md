deep_boltzmann
==============
This class implements a deep boltzmann machine as described by [Hinton and Salakhutdinov][1], as well as a dropout back-propagation for some reasonable definition thereof. It does both of these with stochastic gradient descent.

The whole thing, is, of course, just a prototype, with no guarantees of anything, but it seems to work. The design is based heavily on use of numpy.einsum.


[1]: https://www.cs.toronto.edu/~hinton/absps/efficientDBM.pdf
