from dbm import DBM    
import numpy as np
import joblib


def render_output(i,k):
    energy.append(dbm_test.total_energy())
    print 'training layer ',k, ' cycle energy: ', energy[-1]
    joblib.dump(dbm_test, 'output/dbm_test')
    joblib.dump(energy, 'output/dbm_energy')
    W = dbm_test.layers[k]['W']
    print '\t',W.max(), W.min(), W.mean(), W.std()
    bias = dbm_test.layers[k]['bias']
    print '\t', bias.max(), bias.min(), bias.mean(), bias.std()
    if i%10==0:
        data = np.array([[1,1,1]])
        mask = np.array([[1,0,0]])
        print 'target=[1,1,1] (mask [1,0,0]), prediction = ',np.mean(dbm_test.gibbs_predict(data,mask, samples = 100, gibbs_iterations= 10, stop_layer=k),axis=0), 'predict_2', dbm_test.deterministic_predict(data,mask,stop_layer=k)

dataset = np.round(np.random.rand(1000000, 1))
dataset = np.append(dataset,dataset,axis=1)
dataset = np.append(dataset,np.ones((dataset.shape[0],1)),axis=1)
print 'dataset shape: ', dataset.shape


energy = []
entropy = []
accuracy = []
print 'initializing model'
dbm_test=DBM(dataset,layers=[30, 20,10])
render_output(1,1)

for k in range(1,3):
    for i in range(51):
        print 'beginning boltzmann training of model'
        dbm_test.train_unsupervised(k)
        render_output(i,k)
