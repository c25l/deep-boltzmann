import numpy
class DBM(object):
    # dataset: a binary valued data matrix
    # labels: the associated outputs for each data row
    # layers: a list containing the size of each hidden layer
    # fantasy_count: The number of markov chains to run in the background
    # learning_rate: starting learning rate. Will be continued harmonically from the starting value.
    def __init__(self,dataset,labels=numpy.array([]),
                batch_size = 50,
                layers=[10,2],
                fantasy_count = 10,
                learning_rate = .0001, ):


        self.dataset = dataset
        self.labels = labels
        self.datapts = dataset.shape[0]
        self.batch_size = batch_size
        self.features = dataset.shape[1]
        self.fantasy_count = fantasy_count
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append({
                            'size':self.features,
                            'fantasy': numpy.random.randint(0,2,(fantasy_count,self.features)).astype(float),
                            'mu':0,
                            'bias':self.sigma_inverse(numpy.mean(dataset, axis=0)).reshape(1,self.features),
                            })
        for layer in range(len(layers)):
            self.add_layer(layers[layer])
        
        
    
    #Stochastic annealing scheduler. This one assures that, regardless of
    #  starting value the sequence is in l^2-l^1
    def next_learning_rate(self, rate):
        return 1.0/(1.0/rate+1)
    
    
    def l2_pressure(self,weights):
        norms = numpy.sqrt(numpy.sum(weights*weights, axis=0))
        norms = numpy.floor(1/norms.reshape(norms.shape[0],1))
        norms = norms.T.repeat(weights.shape[0], axis=0)
        out = norms*weights
        return -.01*out

    #some sigmoid function, this one is fine.
    def sigma(self, x):
        x = numpy.clip(x,-100,100)
        return 1/(1+numpy.exp(-x))
    def sigma_inverse(self,x):
        x = numpy.clip(x,.00001,.99999)
        return numpy.log(x/(1-x))
    
    #Quick and dirty bootstrapper to manage samples per epoch
    def data_sample(self, num):
        if self.labels.shape[0] > 0:
            return (self.dataset[numpy.random.randint(0, self.dataset.shape[0], num)],
                self.labels[numpy.random.randint(0, self.labels.shape[0], num)])
        else:
            return (self.dataset[numpy.random.randint(0, self.dataset.shape[0], num)],
                numpy.array([]))


    #Returns activations for probabilities, do not use sigma here, because sampling probs directly.
    def sample(self, fn, args):
        temp = fn(*args)
        temp_cutoff = numpy.random.rand(*temp.shape)
        return (temp >temp_cutoff).astype(float)
    

    #This propagates the test state through the net, 
    #does sigmoid at each layer, and passes that along. 
    # Returns probs at the end, because why not.
    def predict_probs(self, test, prop_uncertainty=False, omit_layers=0): 
        out = test
        for i in range(1,len(self.layers)-omit_layers):
            W=self.layers[i]['W']
            bias = self.layers[i]['bias']
            out = self._predict(W,bias,out)
            if not prop_uncertainty and i< len(self.layers)-1:
                out =numpy.round(out)
        return out
    
    def _predict(self,W,bias,inputs):
        return self.sigma(bias + numpy.dot(inputs,W))


    #The energy of a given layer with a given input and output vector
    def _energy(self,v,W,h,bv,bh):
        return numpy.mean(-numpy.dot(v,bv.T) -numpy.dot(h,bh.T))- numpy.tensordot(numpy.dot(v,W),h, axes=([0,1],[0,1]))

    
    #The energy of the whole DBM with given inputs and hidden activations
    def internal_energy(self, v, hs):
        temp=self._energy(v, self.layers[1]['W'], hs[0],self.layers[0]['bias'],self.layers[1]['bias'])
        for i in range(1, len(self.layers)-1):
            temp += self._energy(hs[i-1], self.layers[i+1]['W'], hs[i], self.layers[i]['bias'], self.layers[i+1]['bias'])
        return temp

    
    #The energy of the network given only the input activiation.
    def energy(self, v):
        hs =  [numpy.round(self.sigma(self.layers[1]['bias']+numpy.dot(v,self.layers[1]['W'])))]
        for i in range(2,len(self.layers)):
            hs.append(numpy.round(self.sigma(self.layers[i]['bias']+numpy.dot(hs[-1], self.layers[i]['W']))))
        return self.internal_energy(v,tuple(hs))
    

    #return the total energy of the stored dataset 
    #and its activation structure given the current model
    def total_energy(self):
        return self.energy(self.dataset)
    

    #return the total entropy of the dataset given the current model.
    def total_entropy(self):
        pred = numpy.clip(self.predict_probs(self.dataset),0.0001,.9999)
        return numpy.sum(self.labels*numpy.log(pred) + (1-self.labels)*numpy.log(1-pred))
    
    
    # prob_given_vis gives a vector of length j with the corresponding probs
    # subset to theappropriate entry to get hj1==1
    def prob_given_vis(self, W, vs,bias, double=False):
        if double:
            return self.sigma(2*(bias + numpy.dot(vs, W)))
        else:
            return self.sigma(bias + numpy.dot(vs, W))


    #prob_given_out is the same as above, but with the opposite value  and convention.
    def prob_given_out(self, W, hs,bias,double=False):
        if double:
            return self.sigma(2*(bias + numpy.dot(hs, W.T)))
        else:
            return self.sigma(bias + numpy.dot( hs, W.T))


    #Tiny gibbs sampler for the fantasy particle updates. The numer of iterations could be controlled, but needn't be
    def gibbs_update(self, gibbs_iterations=10, layers = None):
        if layers == None:
            layers = len(self.layers)
        for j in range(gibbs_iterations):
            for i in range(1,layers):
                double = bool(i%2) 
                active = self.layers[i-1]['fantasy']
                bias = self.layers[i]['bias']
                W = self.layers[i]['W']
                self.layers[i]['fantasy'] = self.sample(self.prob_given_vis, (W,active,bias,double))
            for i in range(layers-1,1,-1):
                double = not bool(i%2) 
                active = self.layers[i]['fantasy']
                bias = self.layers[i-1]['bias']
                W = self.layers[i]['W']
                self.layers[i-1]['fantasy'] = self.sample(self.prob_given_out,(W,active,bias, double))

            
    #This is stochastic gradient descent version of a dropc back-propagator.
    def dropc_step(self,data,labels,rate, 
                     dropout_fraction = 0, momentum_decay = 0, train_layers=1):
        assert labels != None, 'no labels defined' 
        layers=len(self.layers)
        min = layers-train_layers -1
        for layer in range(layers-1,min,-1):
            W=self.layers[layer]['W']
            dropout = numpy.ones(W.shape)
            #really don't want to drop all the connections. Just in case...
            while numpy.min(dropout) >=1 and dropout_fraction>0:
                dropout = (numpy.random.rand(*W.shape)<dropout_fraction).astype(float)
            self.layers[layer]['dropout array']= dropout
            self.layers[layer]['dropped out'] = W*dropout
            W = W-self.layers[layer]['dropped out']
            self.layers[layer]['W']=W
        for layer in range(layers-1,min,-1):
            act = self.predict_probs(data)
            prior_act = self.predict_probs(data, omit_layers=layers-layer)
            W = self.layers[layer]['W']
            errors = act - labels
            for iter in range(layers-1, layer,-1):
                source = self.layers[iter]
                errors = source['W'].T*errors
            #output layer
            dropout =  self.layers[layer]['dropout array']

            derivative = act * (1-act) * errors
            errors = act * (1-act)*errors
            gradient = 1.0/self.datapts * numpy.dot(prior_act.T,derivative)
            momentum = momentum_decay*self.layers[layer]['momentum']
            gradient = rate * gradient * (1-dropout)
            W = W - gradient - momentum
            self.layers[layer]['momentum'] = momentum + gradient
            self.layers[layer]['W']=W + self.l2_pressure(W)
        for layer in range(layers-1,min,-1):
            W= self.layers[layer]['W']
            W = W+self.layers[layer]['dropped out']  
            self.layers[layer]['W'] = W +0.0001*numpy.random.randn(*W.shape)


    #Train, or continue training the model according to the training schedule for another train_iterations iterations
    def train_unsupervised(self, layer, train_iterations=10000, gibbs_iterations=10):
        layers=len(self.layers)
        for iter in range(train_iterations):
            self.gibbs_update(gibbs_iterations,layer)
            data, labels = self.data_sample(self.batch_size)
            rate = self.learning_rate
            self.learning_rate=self.next_learning_rate(self.learning_rate)
            
            previous = numpy.round(self.predict_probs(data, omit_layers=layers-layer))
            bias = self.layers[layer]['bias']
            mu = bias+numpy.dot(previous,self.layers[layer]['W'])
            #I came up with this bias update scheme. It's not actually
            #in the papers, but it seems reasonable.
            bias_part = mu.mean(axis=0).reshape(*bias.shape)
            self.layers[layer]['bias'] = bias + rate*(bias_part-bias)
            if layer%2==0:
                mu = self.sigma(2*mu)
            else:
                mu = self.sigma(mu)
            self.layers[layer]['mu'] = mu 
            gradient_part = - 1.0/(self.datapts*self.batch_size) * numpy.dot(previous.T, mu)
            approx_part =- 1.0/self.fantasy_count * numpy.dot(self.layers[layer-1]['fantasy'].T,
                                                              self.layers[layer]['fantasy'])
            W =( self.layers[layer]['W'] 
                + rate *gradient_part 
                + rate *approx_part)
            self.layers[layer]['W'] = W + self.l2_pressure(W)

    

    #Assuming the data came in with labels, which were disregarded during the unsupervised training.
    def train_dropc(self, train_iterations=10000, weight=1, layers = 1):
        for iter in range(train_iterations):
            rate = self.learning_rate
            rows, labels = self.data_sample(1)
            self.dropc_step(rows, labels, self.learning_rate, rate*weight, layers)               
        self.learning_rate=self.next_learning_rate(self.learning_rate)

    #Okay, so this is an attempt at prediction using a gibbs sampling technique. 
    #The idea is that you feed in an input, but
    #this input is incomplete. You want to make it complete by using 
    #the information in the network, so you update the network, 
    #and sample repeatedly, keeping in mind that the values you want are going 
    #to be set by the mask(==1) and the unknowns will be in flux.
    #the averages of the output values should tell you something.
    #If the mask is none, it will just make up data given your inputs.
    def gibbs_predict(self, input, mask=None,samples = 100,  gibbs_iterations=100, stop_layer = None):
        layers = len(self.layers)
        input_state = {0:input}
        if stop_layer is None:
            stop_layer = layers-3
        for i in range(1,stop_layer+2):
            input_state[i] = numpy.zeros((input.shape[0],self.layers[i]['W'].shape[1]))
        out = []
        for j in range(gibbs_iterations*samples):
            for i in range(1,stop_layer-1):
                active = input_state[i-1]
                bias = self.layers[i]['bias']
                W = self.layers[i]['W']
                input_state[i] = self.sample(self.prob_given_vis, (W,active,bias))
            for i in range(stop_layer-2,0,-1):
                active = input_state[i+1]
                bias = self.layers[i]['bias']
                W = self.layers[i+1]['W']
                input_state[i] = self.sample(self.prob_given_out,(W,active,bias))

            candidate = self.sample(self.prob_given_out, (self.layers[1]['W'], input_state[1],self.layers[0]['bias']))
            if mask is not None:
                input_state[0] = candidate*(1-mask) + input_state[0]*mask 
            else:
                input_state[0]=candidate
            if j%gibbs_iterations == gibbs_iterations-1:
                out.append(input_state[0])
        return out

    #This detpred method sounds the inputs into the DBM and reads the echo
    #from the back of the network. Definitely not the way to do this, but its'
    #faster and conceptually cheaper than the AIS methods.
    def deterministic_predict(self, input, mask=None,  stop_layer = None):
        layers = len(self.layers)
        input_state = {0:input}
        if stop_layer is None:
            stop_layer = layers-3
        for i in range(1,stop_layer+2):
            input_state[i] = numpy.zeros((input.shape[0],self.layers[i]['W'].shape[1]))
        for i in range(1,stop_layer-1):
            active = input_state[i-1]
            bias = self.layers[i]['bias']
            W = self.layers[i]['W']
            input_state[i] = numpy.round(self.prob_given_vis(W,active,bias))
        for i in range(stop_layer-2,0,-1):
            active = input_state[i+1]
            bias = self.layers[i]['bias']
            W = self.layers[i+1]['W']
            input_state[i] = numpy.round(self.prob_given_out(W,active,bias))

        candidate = self.prob_given_out(self.layers[1]['W'], input_state[1],self.layers[0]['bias'])
        if mask is not None:
            input_state[0] = candidate*(1-mask) + input_state[0]*mask 
        else:
            input_state[0]=candidate
        return input_state[0]

    
    #append new layers to existing objects!
    def add_layer(self, size): 
        hidden = {'size':size, 'mu':0}
        above = self.layers[-1]['size']
        hidden['W'] = numpy.random.randn(above,size)
        hidden['bias'] = numpy.random.randn(1,size)
        hidden['momentum'] = numpy.zeros((above,size))
        hidden['fantasy'] = numpy.random.randint(0,2,(self.fantasy_count,size)).astype(float)
        self.layers.append(hidden)
           
