import numpy
class DBM(object):
    # dataset: a binary valued data matrix
    # labels: the associated outputs for each data row
    # layers: a list containing the size of each hidden layer
    # fantasy_count: The number of markov chains to run in the background
    # learning_rate: starting learning rate. Will be continued harmonically from the starting value.
    def __init__(self,dataset,labels,
                batch_size = 100,
                layers=[10,2],
                fantasy_count = 100,
                learning_rate = .001, ):

        self.dataset = dataset
        self.labels = labels
        self.datapts = dataset.shape[0]
        self.batch_size = batch_size
        self.features = dataset.shape[1]
        self.fantasy_count = fantasy_count
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append({'layer':'visible',
                            'size':self.features,
                            'fantasy': numpy.random.randint(0,2,(fantasy_count,self.features)).astype(float),
                            'mu':0})
        for layer in range(len(layers)):
            size = layers[layer]
            hidden = {'layer':'hidden '+str(layer), 'size':size, 'mu':0}
            above = self.layers[-1]['size']
            hidden['W'] = numpy.random.randn(above,size)
            hidden['momentum'] = numpy.zeros((above,size))
            hidden['fantasy'] = numpy.random.randint(0,2,(fantasy_count,size)).astype(float)
            self.layers.append(hidden)
        
    
    #Stochastic annealing scheduler. This one assures that, regardless of
    #  starting value the sequence is in l^2-l^1
    def next_learning_rate(self, rate):
        return 1.0/(1.0/rate+1)
    
    #calculate the row-wise norm of a matrix, returning a vector whose elements are the row's norms. 
    def normalize(self, weights):
        norms = 1/(numpy.sqrt(numpy.sum(weights*weights, axis=0))+.000001)
        norms = norms.reshape(norms.shape[0],1)
        norms = numpy.clip(norms,0,1)
        norms = norms.reshape(1,norms.shape[0]).repeat(weights.shape[0], axis=0)
        return weights*norms

        
    #some sigmoid function, this one is fine.
    def sigma(self, x):
        return 1/(1+numpy.exp(-x))

    def d_sigma(self, x):
        return self.sigma(x)*self.sigma(1-x)
    
    def sigma_inverse(self, x):
        x = numpy.clip(x, .0001,.9999)
        return numpy.log(x/(1-x))

    #Quick and dirty bootstrapper to manage samples per epoch
    def data_sample(self, num):
        return (self.dataset[numpy.random.randint(0, self.dataset.shape[0], num)],
                self.labels[numpy.random.randint(0, self.labels.shape[0], num)])


    #Returns activations for probabilities, do not use sigma here, because sampling probs directly.
    def sample(self, fn, args):
        temp = fn(*args)
        temp_cutoff = numpy.random.rand(*temp.shape)
        return (temp>temp_cutoff).astype(float)
    

    #This propagates the test state through the net, does sigmoid at each layer, and passes that along. 
    # Returns probs at the end, because why not.
    def predict_probs(self, test, prop_uncertainty=False, omit_layers=0): 
        out = test
        for i in range(1,len(self.layers)-omit_layers):
            W=self.layers[i]['W']
            out = self._predict_stage_probs(W,out)
            if not prop_uncertainty and i< len(self.layers)-1:
                out =numpy.round(out)
        return out
    
    def _predict_stage_probs(self,W,inputs):
        return self.sigma(numpy.dot(inputs,W))


    #The energy of a given layer with a given input and output vector
    def _energy(self,v,W,h):
        return -numpy.tensordot(numpy.dot(v,W),h, axes=([0,1],[0,1]))

    
    #The energy of the whole DBM with given inputs and hidden activations
    def internal_energy(self, v, hs):
        temp=self._energy(v, self.layers[1]['W'], hs[0])
        for i in range(1, len(self.layers)-1):
            temp += self._energy(hs[i-1], self.layers[i+1]['W'], hs[i])
        return temp

    
    #The energy of the network given only the input activiation.
    def energy(self, v):
        hs =  [numpy.round(self.sigma(numpy.dot(v,self.layers[1]['W'])))]
        for i in range(2,len(self.layers)):
            hs.append(numpy.round(self.sigma(numpy.dot(hs[-1], self.layers[i]['W']))))
        return self.internal_energy(v,tuple(hs))
    

    #return the total energy of the stored dataset and its activation structure given the current model
    def total_energy(self):
        return self.energy(self.dataset)
    

    #return the total entropy of the dataset given the current model.
    def total_entropy(self):
        pred = self.predict_probs(self.dataset)
        return numpy.sum(self.labels*numpy.log(pred) + (1-self.labels)*numpy.log(1-pred))
    
    
    # prob_given_vis gives a vector of length j with the corresponding probs
    # subset to theappropriate entry to get hj1==1
    def prob_given_vis(self, W, vs):
        return self.sigma(numpy.dot(vs, W))


    #prob_given_out is the same as above, but with the opposite value  and convention.
    def prob_given_out(self, W, hs):
        return self.sigma(numpy.dot( hs, W.T))


    #for deeper nets, you need the above-and-below layer contributions. This aggregates both1
    def prob_internal(self, W0, W1, vs, h2):
        return self.sigma(numpy.dot(vs, W0)+numpy.dot(h2,  W1.T))


    #Tiny gibbs sampler for the fantasy particle updates. The numer of iterations could be controlled, but needn't be
    def gibbs_update(self, gibbs_iterations=100):
        for j in range(gibbs_iterations):
            self.layers[0]['fantasy'] = self.sample(self.prob_given_out, (self.layers[1]['W'], self.layers[1]['fantasy']))
            self.layers[-1]['fantasy'] = self.sample(self.prob_given_vis, (self.layers[-1]['W'], self.layers[-2]['fantasy']))
            for i in range(1,len(self.layers)-1):
                self.layers[i]['fantasy'] = self.sample(self.prob_internal, (self.layers[i]['W'],
                                                                       self.layers[i+1]['W'], 
                                                                       self.layers[i-1]['fantasy'], 
                                                                       self.layers[i+1]['fantasy'],))



    #backpropagate label assignments to previous layer.       
    def backprop_label(self, label, layers=0):
        end = len(self.layers)-1
        out = label
        for i in range(min(layers,end)):
            W = self.layers[end-i]['W']
            out = numpy.dot(self.sigma_inverse(out), W.T)
        return out

    #This step does the boltzmann part.
    def unsupervised_step(self, data, labels,rate):
        layers=len(self.layers)
        # You could train the last layer too, 
        # but as the last layer communicates the
        # supervised results, this might not help you 
        # so much as just ruin all predictions.
        for i in range(1,layers):
            if i==1:
                previous = data
            else:
                previous = self.layers[i-1]['mu']
            self.layers[i]['mu'] = self.sigma(numpy.dot(previous,self.layers[i]['W']))/data.shape[0]
            
            gradient_part = numpy.dot(previous.T, 
                                      self.layers[i]['mu'])
            approx_part = -1.0/self.fantasy_count * numpy.dot(self.layers[i-1]['fantasy'].T,
                                                                 self.layers[i]['fantasy'])
            self.layers[i]['W'] = self.normalize(self.layers[i]['W'] 
                                                 + rate *(gradient_part 
                                                 + approx_part))


    #This is stochastic gradient descent version of a dropout back-propagator.
    def dropout_step(self,data,labels,rate,weight, dropout_fraction = .5, momentum_decay = 0.25250):
        
        layers=len(self.layers)
        scale_factor = 1.0
        for layer in range(layers-1,0,-1):
            W=self.layers[layer]['W']
            dropout = numpy.zeros(W.shape)
            while numpy.min(dropout) >=1:
                dropout = (numpy.random.rand(*W.shape)<dropout_fraction).astype(float)
            self.layers[layer]['dropout array']= dropout
            self.layers[layer]['dropped out'] = W*dropout
            W = W-self.layers[layer]['dropped out']
            self.layers[layer]['W']=W
            
        for layer in range(layers-1,0,-1):
            prop_label=labels
            if layer < layers-1:
                prop_label = numpy.round(self.sigma(self.backprop_label(labels, layers-1-layer)))
            act = self.predict_probs(data, omit_layers=layers-layer-1)
            prior_act = self.predict_probs(data, omit_layers=layers-layer)
            W = self.layers[layer]['W']
            #output layer
            dropout =  self.layers[layer]['dropout array']
            difference = act - prop_label
            d_activated = self.d_sigma(self.sigma_inverse(act))
            
            #Aiming to for deviance.
            difference = numpy.clip(difference,0.001,.9999)
            gradient = ((-numpy.dot(prior_act.T,1/difference*d_activated)) 
                          + numpy.dot(1-prior_act.T,1/(difference)*d_activated))
            #scale_factor = numpy.sqrt(numpy.sum(1-dropout))*scale_factor
            momentum = self.layers[layer]['momentum']
            gradient_term = weight * gradient / scale_factor * (1-dropout)
            W = W - gradient_term  -  momentum*momentum_decay
            self.layers[layer]['momentum'] = momentum*momentum_decay + gradient_term
            self.layers[layer]['W']=W
            
        for layer in range(layers-1,0,-1):
            W=self.normalize(self.layers[layer]['W'])
            W = W+self.layers[layer]['dropped out'] 
            self.layers[layer]['W'] = W+0.0001*numpy.random.randn(*W.shape)


    #Train, or continue training the model according to the training schedule for another train_iterations iterations
    def train_unsupervised(self, train_iterations=1000, gibbs_iterations=10):
        for iter in range(train_iterations):
            self.gibbs_update(gibbs_iterations)
            data, labels = self.data_sample(self.batch_size)
            rate = self.learning_rate
            self.unsupervised_step(data,labels,rate)
            self.learning_rate=self.next_learning_rate(self.learning_rate)


    #Assuming the data came in with labels, which were disregarded during the unsupervised training.
    def train_supervised(self, train_iterations=1000, weight=.01):
        layers=len(self.layers)
        for iter in range(train_iterations):
            rows, labels = self.data_sample(self.batch_size)
            self.dropout_step(rows, labels, self.learning_rate, weight, dropout_fraction=0.0)

    
    #Assuming the data came in with labels, which were disregarded during the unsupervised training.
    def train_dropout(self, train_iterations=1000, weight=1):
        layers=len(self.layers)
        for iter in range(train_iterations):
            rate = self.learning_rate
            rows, labels = self.data_sample(self.batch_size)
            self.dropout_step(rows, labels, self.learning_rate, rate*weight)               
        self.learning_rate=self.next_learning_rate(self.learning_rate)


 
    #Alternate boltzmann and backprop steps, this could be better than doing a lot of both, 
    #as it helps to co-optimize the energy and entropy.
    def train_hybrid(self, train_iterations=1000, gibbs_iterations = 100 ):
        for iter in range(train_iterations):
            self.gibbs_update(gibbs_iterations)
            data, labels = self.data_sample(self.batch_size)
            rate = self.learning_rate
            self.unsupervised_step(data,labels,rate)
            self.dropout_step(data,labels,rate, 10*rate)   
        self.learning_rate=self.next_learning_rate(self.learning_rate)

