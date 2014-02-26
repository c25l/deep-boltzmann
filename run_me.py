from dbm import DBM    
import numpy as np
import joblib


def render_output():
    energy.append(dbm_test.total_energy())
    print 'post-training cycle energy: ', energy[-1]
    joblib.dump(dbm_test, 'dbm_test')
    joblib.dump(energy, 'dbm_energy')


dataset = np.round(np.random.rand(100000, 3))
dataset = np.repeat(dataset,2,axis=1)
dataset = np.append(dataset,np.ones((dataset.shape[0],1)),axis=1)
print 'dataset shape: ', dataset.shape


energy = []
entropy = []
accuracy = []
print 'initializing model'
dbm_test=DBM(dataset,layers=[8,6])
render_output()

for i in range(100000):
    print 'beginning boltzmann training of model'
    dbm_test.train_unsupervised()
    render_output()
