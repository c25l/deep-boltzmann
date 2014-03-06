from dbm import DBM    
import numpy as np
import joblib


def render_output(i,k):
    energy.append(dbm_test.total_energy())
    print i,'cycle, layer ',k, ' cycle energy: ', energy[-1]
    joblib.dump(dbm_test, 'output/dbm_test')
    joblib.dump(energy, 'output/dbm_energy')

def render_supervised(i):
    entropy.append(dbm_test.total_entropy())
    accuracy.append(1-np.mean(np.abs(np.round(dbm_test.predict_probs(dataset))-labels)))
    print i, ' cycle entropy: ', entropy[-1],' cycle accuracy: ', accuracy[-1]
    joblib.dump(entropy, 'output/dbm_entropy')
    joblib.dump(accuracy, 'output/dbm_accuracy')

dataset = np.round(np.random.rand(10000, 1))
labels = 1-dataset
dataset = np.append(dataset,1-dataset,axis=1)
dataset = np.append(dataset,np.ones((dataset.shape[0],1)),axis=1)
print 'dataset shape: ', dataset.shape


energy = []
entropy = []
accuracy = []
print 'initializing model'
dbm_test=DBM(dataset,layers=[])#[30, 20,10])
#render_output(1,1)

for k in range(1,4):
    for i in range(0):
        print 'beginning boltzmann training of model'
        dbm_test.train_unsupervised(k)
        render_output(i,k)

dbm_test.learning_rate = 1
dbm_test.add_layer(1)
dbm_test.labels = labels
#Adapt the output layer to the network
for i in range(100):
    #train backprop
    dbm_test.train_backprop(layers=1)
    render_output(i,4)
    render_supervised(i)

#Train the whole thing towards a minimum.
for i in range(100):
    #train backprop
    dbm_test.train_backprop()
    render_output(i,4)
    render_supervised(i)
