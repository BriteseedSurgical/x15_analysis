import torch

import numpy as np

loss_nan_ctr = 0
op_nan_ctr   = 0

#Extract data from file , tensorize , make dataset. Can I 
#easily use the batch output of a dataset ? 

def pick_examples(X , Y , batch_size = 32):
    x_list = None
    y_list = None
    nx = len(X)
    ny = len(Y)
    x_indices = np.random.randint(0,nx,batch_size)
    y_indices = np.random.randint(0,ny,batch_size)
    x_list = [X[x_indices[j]] for j in range(0,batch_size)]
    y_list = [Y[y_indices[j]] for j in range(0,batch_size)]
    #random_indices = make a set of random indices
    #Make lists with those data points 
    return [x_list , y_list]

def generate_pairs(X , Y , batch_size = 32):
    XY = []
    T  = []
    #what is the best way to generate this ? 
    #Will this be the performance bottleneck?
    #If 1000 examples each , it's 3 GB of data -> unreasonable.
    #Generate dynamically , at random , and generate long enough to statistically equalize.
    
    #Generate X and Y randomly (assume , done outside)

    #Here , do
    #Let X and Y have lengths of powers of 2
    #Combine first half of x with it's second half -> assign target 1 (similar)
    #Combine first half of y with it's second half -> assign target 1 (similar)
    #Combine index wise , X and Y                  -> assign target 0 (dissimilar)
    #Push to XY , each (x,y)->t from each of the above three cases.
    
    #Tensorize here                                           ? -> Yes for now.
    #Make sure number of positive and negative pairs are equal? -> Yes for now.

    divide_point = int(0.5 * batch_size)
    #Do I want the inputs as lists or arrays ? Lists for now...
    for j in range(0,divide_point):
        XY.append([X[j] , X[divide_point+j]])
        T.append(int(1))
    for j in range(0,batch_size):
        XY.append([X[j],Y[j]])
        T.append(int(-1))
    for j in range(0,divide_point):
        XY.append([Y[j] , Y[divide_point+j]])
        T.append(int(1))

    #Send out tensors ? What data type does the cosine loss seek for targets ?
    #It's a float target.
    XY = torch.Tensor(XY).float()
    T  = torch.Tensor(T).float()

    return [XY,T]

def train_step_contrastive(system , x , y , t , objective , optimizer):
    
    global loss_nan_ctr , op_nan_ctr
    
    loss     = None
    accuracy = None

    embedding_x = system(x)
    embedding_y = system(y)

    if torch.isnan(embedding_x).any() or torch.isnan(embedding_y).any():
        
        op_nan_ctr = op_nan_ctr + 1
        print('Case of output == nan.')

    loss = objective(embedding_x,embedding_y,t) #If cosine embedding , t == -1 or 1

    if torch.isnan(loss):
        loss_nan_ctr = loss_nan_ctr + 1
        print('Skipping backprop.')
    else:
        loss.backward()
        optimizer.step()

    #compute accuracy as number of cases greater than and lesser than zero.
    
    return [loss , accuracy]

#x and y batch sizes
#xy batch size (2*x batch size or 2*y batch size)

#Main training loop
#Main training loop

#Curious to look at the model summary - > And the combined system graph