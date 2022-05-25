import numpy as np
import scipy.io as sio

import os
import json

if __name__ == '__main__':
    
    root   = os.getcwd()
    target_name = 'V9_native'
    target = os.path.join(root,target_name + '.mat')

    all_data = sio.loadmat(target)
    examples = all_data['XTrain']
    truths   = all_data['YTrain']

    n_examples = np.shape(examples)[0]
    
    example_list = []

    for i in range(0,n_examples):
        example_list.append(examples[i,0])
    example_list = np.array(np.stack(example_list))
    truths       = np.squeeze(truths)
    example_list = example_list[:,:,:,np.newaxis]

    np.save('xtrain_' + target_name + '.npy' , example_list)
    np.save('ytrain_' + target_name + '.npy' , truths)
