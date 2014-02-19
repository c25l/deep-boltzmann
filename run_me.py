from dbm import DBM    
import numpy as np

np.random.seed(1)
test = np.random.rand(2,10)
label =np.round(np.transpose(np.transpose(test)[1:3]).reshape(2,2))
print 'test shape: ', test.shape, 'label shape: ', label.shape

dataset  = np.random.rand(100000, 10)
labels = np.round(np.transpose(np.transpose(dataset)[1:3]).reshape(100000,2))
print 'dataset shape: ', dataset.shape, 'labels shape: ', labels.shape

print 'initializing model'
dbm_test=DBM(dataset,labels)
print 'test predictions and labels; ', dbm_test.predict_probs(test).ravel(), label.ravel()
print 'dataset energy and entropy: ', dbm_test.total_energy(), dbm_test.total_entropy()

for i in range(2):
    print 'beginning unsupervised training of model'
    dbm_test.train_unsupervised()
    print 'test predictions and labels: ', dbm_test.predict_probs(test).ravel() ,label.ravel()
    print 'post-training cycle energy and entropy: ',dbm_test.total_energy(), dbm_test.total_entropy()

for i in range(100):
    print 'beginning dropout training of model'
    dbm_test.train_dropout(weight=11)
    print 'test predictions and labels: ', dbm_test.predict_probs(test).ravel() ,label.ravel()
    print 'post-training cycle energy and entropy: ',dbm_test.total_energy(), dbm_test.total_entropy()
