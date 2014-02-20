from dbm import DBM    
import numpy as np

np.random.seed(1)
test = np.random.rand(2,3)
label =np.round(np.transpose(np.transpose(test)[1]).reshape(2,1))
print 'test shape: ', test.shape, 'label shape: ', label.shape
dataset  = np.random.rand(100000, 3)
labels = np.round(np.transpose(np.transpose(dataset)[1]).reshape(100000,1))
print 'dataset shape: ', dataset.shape, 'labels shape: ', labels.shape

print 'initializing model'
dbm_test=DBM(dataset,labels, layers=[6,1])
print 'test predictions and labels; ', dbm_test.predict_probs(test).ravel(), label.ravel()
print 'dataset energy and entropy: ', dbm_test.total_energy(), dbm_test.total_entropy()

for i in range(10):
    print 'beginning unsupervised training of model'
    dbm_test.train_unsupervised()

    print 'test predictions and labels: ', dbm_test.predict_probs(test).ravel() ,label.ravel()
    print 'post-training cycle energy and entropy: ',dbm_test.total_energy(), dbm_test.total_entropy()
print dbm_test.layers[1]['W']
print dbm_test.layers[-1]['W']
dbm_test.learning_rate=1
for i in range(1000):
    print 'beginning dropout training of model'
    dbm_test.train_dropout(weight = .01)

    print 'test predictions and labels: ', dbm_test.predict_probs(test).ravel() ,label.ravel()
    print 'post-training cycle energy and entropy: ',dbm_test.total_energy(), dbm_test.total_entropy()
print dbm_test.layers[1]['W']
print dbm_test.layers[-1]['W']
