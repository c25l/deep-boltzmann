from dbm import DBM    
import numpy as np
import joblib


def render_output():
    entropy.append( dbm_test.total_entropy())
    energy.append(dbm_test.total_energy())
    prediction = dbm_test.predict_probs(dataset)
    accuracy.append(1-np.abs(np.round(prediction)-labels).sum()/labels.shape[0])
    print 'test predictions and labels; ', dbm_test.predict_probs(test).ravel(), label.ravel()
    print 'post-training cycle energy, entropy, and accuracy: ', energy[-1], entropy[-1], accuracy[-1]
    print 'max prediction: ', np.max(prediction), ' min prediction: ', np.min(prediction)
    joblib.dump(dbm_test, 'dbm_test')
    joblib.dump(energy, 'dbm_energy')
    joblib.dump(entropy, 'dbm_entropy')
    joblib.dump(accuracy, 'dbm_accuracy')
    print dbm_test.layers[1]['W']
    print dbm_test.layers[2]['W']


np.random.seed(1)
test = np.random.rand(2,3)
label =np.round(np.transpose(np.transpose(test)[1]).reshape(2,1))
print 'test shape: ', test.shape, 'label shape: ', label.shape
dataset  = np.random.rand(100000, 3)
labels = np.round(np.transpose(np.transpose(dataset)[1]).reshape(100000,1))
print 'dataset shape: ', dataset.shape, 'labels shape: ', labels.shape


energy = []
entropy = []
accuracy = []
print 'initializing model'
dbm_test=DBM(dataset,labels, layers=[8,1], batch_size=100)
render_output()

for i in range(2):
    for i in range(0):
        print 'beginning dropout training of model'
        dbm_test.train_dropout(weight = 0.1)
        render_output()


    for i in range(5):
        print 'beginning boltzmann training of model'
        dbm_test.train_unsupervised()
        render_output()
dbm_test.learning_rate=1.0

for i in range(0):
    print 'beginning hybrid training of model'
    dbm_test.train_hybrid()
    render_output()
dbm_test.learning_rate=1.0

for i in range(1000):
    print 'beginning dropout training of model'
    dbm_test.train_dropout(weight = 0.001)
    render_output()


