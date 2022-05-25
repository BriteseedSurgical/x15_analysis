import os
import json
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import time
from x15_dl_spawn import *
from pathlib import Path

from tqdm import tqdm

CMOS_MAX = 4096

loss_nan_ctr = 0
op_nan_ctr   = 0

def train_step_contrastive(system , x , y , t , objective , optimizer):
    
    global loss_nan_ctr , op_nan_ctr
    
    loss     = None

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
    
    return [loss , embedding_x , embedding_y]

def eval_step_contrastive(system , x, y , t , distance_measure , th = 0):
    eval_loss     = 0
    eval_accuracy = 0
    t = t.detach().numpy()
    system.eval()
    with torch.no_grad():
        x_embeddings = system(x)
        y_embeddings = system(y)
        distances    = distance_measure(x_embeddings,y_embeddings)
        distances    = distances.detach().numpy()

        for j in range(0,np.shape(distances)[0]):
            if distances[j] > 0 and t[j] == 1:
                eval_accuracy = eval_accuracy + 1
            elif distances[j] < 0 and t[j] == -1:
                eval_accuracy = eval_accuracy + 1
            else:
                _ = 0
        
        eval_accuracy = (eval_accuracy/np.shape(distances)[0]) * 100
        #print('Eval accuracy : ' , eval_accuracy)


    return [eval_accuracy , eval_loss]

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
    #XY = []
    Ax = []
    Bx = []
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
        #XY.append([X[j] , X[divide_point+j]])
        Ax.append(X[j])
        Bx.append(X[divide_point+j])
        T.append(1)
    for j in range(0,batch_size):
        #XY.append([X[j],Y[j]])
        Ax.append(X[j])
        Bx.append(Y[j])
        T.append(-1)
    for j in range(0,divide_point):
        #XY.append([Y[j] , Y[divide_point+j]])
        Ax.append(Y[j])
        Bx.append(Y[divide_point+j])
        T.append(1)

    Ax = np.array(Ax)
    Bx = np.array(Bx)
    T  = np.array(T)
    
    #Keep X and Y separate -> can save processing time ? 

    #Send out tensors ? What data type does the cosine loss seek for targets ?
    #It's a float target.
    #XY = torch.Tensor(XY).float()
    Ax = torch.Tensor(Ax).float()
    Bx = torch.Tensor(Bx).float()
    T  = torch.Tensor(T).float()

    return [Ax,Bx,T]

classes = ['same' , 'different']
configs = [['ureter','peritoneum'],['peritoneum','none']]

pick_rand  = False
n_cmos     = 99
n_leds     = 6
n_channels = 4
n_sequence = 1

MODE         = 'train'

SEED        = True
SEED_VALUE  = 199
if SEED:
    seed_value = SEED_VALUE

n_cycles               = 32000 #Calculate from total possible pairs.
n_va_cycles            = 256
batch_size             = 128
optimizer_update_epoch = int(n_cycles/2)
starting_lr            = 6e-05
fine_tuning_lr         = 5e-06
momentum               = 0.9
weight_decay_rate      = 2e-05
print_status_frequency = 32 #in epochs

#create a model instance for training. 
primary = x4_def_pretrain() 
#save the initial state as a pt into wandb.

# system summary and i/p o/p test.
system_ip_shape = (n_sequence,n_channels,n_cmos)

cmos_proxy = np.random.uniform(0,1, size = (batch_size,n_sequence,n_channels,n_cmos))
cmos_proxy = torch.from_numpy(cmos_proxy).float()

example_ip = cmos_proxy

device = None
if torch.cuda.is_available():
    print("CUDA device found. System pushed to device.")
    device = 'cuda'
    primary.cuda()
    example_ip.cuda()
else:
    print("CUDA capable device not found. System stays with host.")
    device = 'cpu'

print("System summary.")
summary(primary, input_size = system_ip_shape)
print("")
# system summary and i/p o/p test.

optimizable_parameters = primary.parameters()
optimizer              = optim.Adam(optimizable_parameters, lr = starting_lr, weight_decay = weight_decay_rate)
criterion              = nn.CosineEmbeddingLoss()
val_criterion          = nn.CosineSimilarity()

if __name__ == '__main__':
    
    print('')
    print('Torch version : ' , torch.__version__)
    print('')

    system_ops    = []
    tr_losses     = []

    if SEED:
        print("Setting seeds...")
        
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    else:
        print("Unseeded run ...")


    root             = os.getcwd()
    parent           = Path(root)
    parent           = parent.parent.absolute()

    x15_dl_data_main = os.path.join(parent , 'x15_dl')
    pretraining_main = os.path.join(x15_dl_data_main, 'unsupervised')
    if pick_rand:
        data_at      = os.path.join(pretraining_main , 'x15_x4_ut.json')
    else:
        data_at      = os.path.join(pretraining_main , 'x15_x4_ut_unrand.json')   
    file_pointer     = open(data_at)
    content          = json.load(file_pointer)
    xt               = json.loads(content)
        
    x_train_class1   = xt[configs[0][0] + '_' + configs[0][1] + '_tr']['x4_cs']
    x_train_class2   = xt[configs[1][0] + '_' + configs[1][1] + '_tr']['x4_cs']

    x_train_class1   = [np.array(x_train_class1[j])/CMOS_MAX for j in range(0,len(x_train_class1))]
    x_train_class2   = [np.array(x_train_class2[j])/CMOS_MAX for j in range(0,len(x_train_class2))]

    n_ex_class1      = len(x_train_class1)
    n_ex_class2      = len(x_train_class2)

    print(np.shape(x_train_class1) , n_ex_class1)
    print(np.shape(x_train_class2) , n_ex_class2)

    x_va_class1   = xt[configs[0][0] + '_' + configs[0][1] + '_va']['x4_cs']
    x_va_class2   = xt[configs[1][0] + '_' + configs[1][1] + '_va']['x4_cs']  

    x_va_class1   = [np.array(x_va_class1[j])/CMOS_MAX for j in range(0,len(x_va_class1))]
    x_va_class2   = [np.array(x_va_class2[j])/CMOS_MAX for j in range(0,len(x_va_class2))] 

    n_va_class1   = len(x_va_class1)
    n_va_class2   = len(x_va_class2)

    print(np.shape(x_va_class1) , n_va_class1)
    print(np.shape(x_va_class2) , n_va_class2)

    print('')
    print('Beginning training for ' + str(n_cycles) + ' cycles , ' + 'with a batch size of ' + \
        str(batch_size) + ' and a training set of ' + str(int(n_ex_class1 * n_ex_class2)) + ' pairs.')
    print('')

    ctr = 0

    for cycle in tqdm(range(n_cycles),desc = 'Training progress'):
                
        if cycle == optimizer_update_epoch:
            print('Finetuning for ' + str(n_cycles - cycle) + ' more epochs with s.g.d at a l.r of ' + str(fine_tuning_lr))
            for g in optimizer.param_groups:
                g['lr'] = fine_tuning_lr
            
        primary.train()
        
        x , y           = pick_examples(x_train_class1 , x_train_class2 , batch_size = int(batch_size/2))
        x , y , truths  = generate_pairs(x , y , batch_size = int(batch_size/2))
        
        x      = x.float()
        y      = y.float()
        truths = truths.float()

        if device == 'cuda':
            x      = x.cuda()
            y      = y.cuda()
            truths = truths.cuda()

        [loss , ex , ey ] = train_step_contrastive(primary , x , y , truths , criterion , optimizer)
        
        if ctr % print_status_frequency == 0:     
            tr_losses.append(loss.item())

        system_ops.append([ex,ey])

        ctr = ctr + 1

    print("")
    print('Training done.')
    print("")
    
    #eval
    ctr = 0
    val_accs = []
    for cycle in range(n_va_cycles):
                        
        x , y           = pick_examples(x_va_class1 , x_va_class2 , batch_size = int(batch_size/2))
        x , y , truths  = generate_pairs(x , y , batch_size = int(batch_size/2))
        
        x      = x.float()
        y      = y.float()
        truths = truths.float()

        if device == 'cuda':
            x      = x.cuda()
            y      = y.cuda()
            truths = truths.cuda()

        [acc_va , loss_va ] = eval_step_contrastive(primary,x,y,truths,val_criterion)
        val_accs.append(acc_va)
        ctr = ctr + 1
    #eval

    print('Average validation accuracy : ' , np.mean(np.array(val_accs)))
    #loop it , and accumulate results over a few batches.annoying.
    #Run validation here.

    plt.figure()
    plt.title( 'Training Loss vs Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(tr_losses)
    plt.show()

    plt.figure()
    plt.title( 'Val Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(val_accs)
    plt.show()
    
    print('Exporting model as an onnx.')
    torch.onnx.export(primary,example_ip,'def_pre_train.onnx') #move to right paths

    print('Writing torch state dict to file. ')
    if pick_rand:
        torch.save(primary.state_dict(),'def_pre_train.pt')
    else:
        torch.save(primary.state_dict(),'def_pre_train_unrand.pt')
    print('')

    print('At main exit line.')