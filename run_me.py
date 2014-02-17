from dbm import DBM    
import numpy as np

test = np.random.randint(0, 2, (2,10))
label =np.transpose(np.transpose(test)[1:3]).reshape(2,2)
print 'test shape: ', test.shape, 'label shape: ', label.shape

dataset  = np.random.randint(0, 2, (100000, 10))
labels = np.transpose(np.transpose(dataset)[1:3]).reshape(100000,2)
print 'dataset shape: ', dataset.shape, 'labels shape: ', labels.shape

print 'initializing model'
dbm_test=DBM(dataset,labels)
print 'test predictions and labels; ', dbm_test.predict_many_probs(test).ravel(), label.ravel()
print 'dataset energy and entropy: ', dbm_test.total_energy(), dbm_test.total_entropy()

print 'beginning hybrid training of model'
dbm_test.train_hybrid()
print 'test predictions and labels: ', dbm_test.predict_many_probs(test).ravel() ,label.ravel()
print 'post-training cycle energy and entropy: ',dbm_test.total_energy(), dbm_test.total_entropy()


