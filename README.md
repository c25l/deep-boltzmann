deep_boltzmann
==============
This class implements a deep boltzmann machine as described by [Hinton and Salakhutdinov][1], as well as back-propagation for some reasonable definition thereof. I have taken liberties with the structure of the algorithm to combine the energy minimization of the boltzmann machine and the entropy maximization of the back propagation together into a back-and-forth algorithm which hopes to make changes to both on smaller timescales. It achieves through stochastic gradient descent.

The whole thing, is, of course, just a prototype, with no guarantees of anything, but it seems to work, and quickly. The design is based heavily on use of numpy.einsum.


[1]: https://www.cs.toronto.edu/~hinton/absps/efficientDBM.pdf
