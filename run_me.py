from dbm import DBM    
import numpy as np
import joblib


def render_output(i):
    energy.append(dbm_test.total_energy())
    print 'post-training cycle energy: ', energy[-1]
    joblib.dump(dbm_test, 'dbm_test')
    joblib.dump(energy, 'dbm_energy')
    if i%10==0:
        data = np.array([[1,1,1]])
        mask = np.array([[1,0,0]])
        print 'target=[1,0,1] (mask [1,0,0]), prediction = ',np.mean(dbm_test.gibbs_predict(data,mask, samples = 100, gibbs_iterations= 10),axis=0)


dataset = np.round(np.random.rand(1000000, 1))
dataset = np.append(dataset,1-dataset,axis=1)
dataset = np.append(dataset,np.ones((dataset.shape[0],1)),axis=1)
print 'dataset shape: ', dataset.shape


energy = []
entropy = []
accuracy = []
print 'initializing model'
dbm_test=DBM(dataset,layers=[20,10])
render_output(1)

for i in range(100000):
    print 'beginning boltzmann training of model'
    dbm_test.train_unsupervised()
    render_output(i)
