"""
Use this script after running nqueens_datagen.py.
nqueens_datagen.py creates a pickle file with desired number of samples.
This script reads the pickle file created by nqueens_datagen and splits it into train val and test sets.
A subset of val set is dumped as a dev set. The dev set is used for early stopping while training the model.
All other hyperparameters are tuned using val set. 
"""


import pickle
from collections import Counter
import numpy as np
import os

np.random.seed(42)
train_size = 50000
val_size = 10000
dev_size = 1000
test_size = 10000
data_dir = 'data'

nq10= pickle.load(open(os.path.join(data_dir,'nqueens_data_10_5.pkl'),'rb'))
train_set = np.random.choice(range(len(nq10)),train_size, False)
remain_set = list(set(range(len(nq10))).difference(set(train_set)))
val_set = np.random.choice(remain_set,val_size,False)
dev_set = np.random.choice(val_set,dev_size,False)
remain_set = list(set(remain_set).difference(set(val_set)))
test_set = np.random.choice(remain_set,test_size,False)
train_data = [nq10[i] for i in train_set]
dev_data = [nq10[i] for i in dev_set]
val_data = [nq10[i] for i in val_set]
test_data = [nq10[i] for i in test_set]
print('Length of train: {}'.format(len(train_data)))
print('Length of dev: {}'.format(len(dev_data)))
print('Length of vall: {}'.format(len(val_data)))
print('Length of test: {}'.format(len(test_data)))
pickle.dump(train_data,open(os.path.join(data_dir,'nqueens_10_5_train{}k.pkl'.format(int(train_size/1000))),'wb'))
pickle.dump(val_data,open(os.path.join(data_dir,'nqueens_10_5_val{}k.pkl'.format(int(val_size/1000))),'wb'))
pickle.dump(test_data,open(os.path.join(data_dir,'nqueens_10_5_test{}k.pkl'.format(int(test_size/1000))),'wb'))
pickle.dump(dev_data,open(os.path.join(data_dir,'nqueens_10_5_dev{}k.pkl'.format(int(dev_size/1000))),'wb'))

print('Train counter:',Counter([x['count'] for x in train_data]))
print('Test counter:',Counter([x['count'] for x in test_data]))
print('Val counter:',Counter([x['count'] for x in val_data]))
print('Dev counter:',Counter([x['count'] for x in dev_data]))

