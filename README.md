deep_boltzmann
==============
This class intends to implement a deep boltzmann machine as described by [Hinton and Salakhutdinov][5]. There could (might (should)) be a better way to get predictions, but a prediction method using gibbs sampling and masks is implemented.


Planned features include: 


1.  [Bias terms][3] (done)
1.  [Better (Actual) Contrastive Divergence][1] (done)
1.  Greedy layerwise pretraining (done)
2.  [AIS estimation of partition function][2]
3.  Use of DBM to initialize a traditional neural network. (done, naively)
4. [DropConnect][4]

It also implements a dropout back-propagation for some reasonable definition thereof using stochastic gradient descent.

The whole thing, is, of course, just a prototype, with no guarantees of anything, but it seems to work. The design is based heavily on use of numpy.einsum.


[1]: https://www.cs.toronto.edu/~hinton/absps/efficientDBM.pdf
[2]: http://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf
[3]: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
[4]: http://cs.nyu.edu/~wanli/dropc/dropc.pdf
[5]: https://www.cs.toronto.edu/~hinton/absps/dbm.pdf
