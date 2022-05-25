import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torchsummary import summary
from   torch.utils.data import TensorDataset, DataLoader

import time
from x15_dl_spawn import *
from pathlib import Path

from tqdm import tqdm


pick_rand        = False
CMOS_MAX         = 4096
loss_nan_ctr     = 0
op_nan_ctr       = 0
validate         = True
LOAD             = True
TRAIN_COMPOSITE  = True


def train_step(system, x , t , objective , optimizer):
    
    global loss_nan_ctr , op_nan_ctr
    loss     = None

    optimizer.zero_grad()
    op          = system(x)

    if torch.isnan(op).any():
        
        op_nan_ctr = op_nan_ctr + 1
        print('Case of output == nan.')

    loss = objective(op,t) 

    if torch.isnan(loss):
        loss_nan_ctr = loss_nan_ctr + 1
        print('Skipping backprop.')
    else:
        loss.backward()
        optimizer.step()

    #compute accuracy as number of cases greater than and lesser than zero.
    
    return [loss , op]

def eval_step(system, x , t , predictor , criterion):
    
    eval_loss     = 0
    eval_accuracy = 0

    rate_0        = 0
    rate_1        = 0

    case_0        = 0
    case_1        = 0

    system.eval()
    with torch.no_grad():

        ops          = system(x)
        preds        = predictor(ops)
        preds        = preds.detach().numpy()

        eval_loss    = criterion(ops,t) 
        t            = t.detach().numpy()

        for j in range(0,np.shape(t)[0]):
            if  (preds[j][0] > preds[j][1]) and t[j] == 0:
                eval_accuracy = eval_accuracy + 1
                rate_0 = rate_0 + 1
                case_0 = case_0 + 1
            elif (preds[j][0] < preds[j][1]) and t[j] == 1:
                eval_accuracy = eval_accuracy + 1
                rate_1 = rate_1 + 1
                case_1 = case_1 + 1
            else:
                if t[j] == 0:
                    case_0 = case_0 + 1
                if t[j] == 1:
                    case_1 = case_1 + 1
        
        eval_accuracy = (eval_accuracy/np.shape(t)[0]) * 100
        if case_0 > 0:
            rate_0 = round((rate_0/case_0)*100,2)
        if case_1 > 0:
            rate_1 = round((rate_1/case_1)*100,2)

    return [eval_accuracy , eval_loss.detach().numpy(), rate_0 , rate_1]

#Write a make_data function

classes = ['ureter' , 'peritoneum']
configs = [['ureter','peritoneum'],['peritoneum','none']]

n_cmos     = 99
n_leds     = 6
n_channels = 4
n_sequence = 1

MODE         = 'train'

SEED        = True
SEED_VALUE  = 199
if SEED:
    seed_value = SEED_VALUE


n_epochs               = 8192
batch_size             = 20
optimizer_update_epoch = int(n_epochs/2)
starting_lr            = 4e-05
fine_tuning_lr         = 5e-06
momentum               = 0.9
weight_decay_rate      = 2e-06
print_status_frequency = 32 #in epochs

if pick_rand:
    path_to_pretrained_weights = 'def_pre_train.pt'
else:
    path_to_pretrained_weights = 'def_pre_train_unrand.pt'

#create a model instance for training. 
f_pre_train = x4_def_pretrain() 
if LOAD:
    f_pre_train.load_state_dict(torch.load(path_to_pretrained_weights))
f_fine_tune = fc_filler(ip_length = 512 , fc_lengths = [256,64,2])
f_composite = nn.Sequential(f_pre_train,f_fine_tune)


config_dict = {}
#model architectures into wandb

dummy_ip = np.random.uniform(0,1, size = (batch_size,1,4,99))
dummy_ip = torch.from_numpy(dummy_ip).float()
torch.onnx.export(f_pre_train,dummy_ip,'pt_default_architecture.onnx') 
#wandb.save('pt_default_architecture.onnx')

dummy_ip = np.random.uniform(0,1, size = (batch_size,512))
dummy_ip = torch.from_numpy(dummy_ip).float()
torch.onnx.export(f_fine_tune,dummy_ip,'ft_default_architecture.onnx') 
#wandb.save('ft_default_architecture.onnx')

#model architectures into wandb

# system summary and i/p o/p test.
system_ip_shape = (n_sequence,n_channels,n_cmos)

cmos_proxy = np.random.uniform(0,1, size = (batch_size,n_sequence,n_channels,n_cmos))
cmos_proxy = torch.from_numpy(cmos_proxy).float()
example_ip = cmos_proxy

device = None
if torch.cuda.is_available():
    print("CUDA device found. System pushed to device.")
    device = 'cuda'
    f_composite.cuda()
    example_ip.cuda()
else:
    print("CUDA capable device not found. System stays with host.")
    device = 'cpu'

print("System summary.")
summary(f_composite, input_size = system_ip_shape)
print("")
# system summary and i/p o/p test.

if TRAIN_COMPOSITE:
    optimizable_parameters = f_composite.parameters()
else:
    optimizable_parameters = f_composite[1].parameters()
optimizer              = optim.Adam(optimizable_parameters, lr = starting_lr, weight_decay = weight_decay_rate)
criterion              = nn.CrossEntropyLoss()
normalizer             = nn.Softmax(dim=1)

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
        data_at          = os.path.join(pretraining_main , 'x15_x4_ut.json')
    else:
        data_at          = os.path.join(pretraining_main , 'x15_x4_ut_unrand.json')
    file_pointer     = open(data_at)
    content          = json.load(file_pointer)
    xt               = json.loads(content)
        
    #make training dataset, dataloader.    
    x_train_class1   = xt[configs[0][0] + '_' + configs[0][1] + '_tr']['x4_cs']
    x_train_class2   = xt[configs[1][0] + '_' + configs[1][1] + '_tr']['x4_cs']

    x_train_class1   = [np.array(x_train_class1[j])/CMOS_MAX for j in range(0,len(x_train_class1))]
    x_train_class2   = [np.array(x_train_class2[j])/CMOS_MAX for j in range(0,len(x_train_class2))]

    n_ex_class1      = len(x_train_class1)
    n_ex_class2      = len(x_train_class2)

    n_training_examples = n_ex_class1 + n_ex_class2
    x_tr = []
    t_tr = []
    for i in  range(0,n_training_examples):
        if i < n_ex_class1:
            x_tr.append(x_train_class1[i])
            t_tr.append(0)
        else:
            x_tr.append(x_train_class2[i-n_ex_class1])
            t_tr.append(1)

    x_tr         = np.array(x_tr)
    t_tr         = np.array(t_tr)

    torch_x      = torch.Tensor(x_tr).float()
    torch_t      = torch.Tensor(t_tr).type(torch.LongTensor)

    train_xt     = TensorDataset(torch_x,torch_t)
    trainloader  = DataLoader(train_xt, batch_size = batch_size , shuffle=True)
    #make training dataset, dataloader.


    #make validation dataset, dataloader.
    x_va_class1   = xt[configs[0][0] + '_' + configs[0][1] + '_va']['x4_cs']
    x_va_class2   = xt[configs[1][0] + '_' + configs[1][1] + '_va']['x4_cs']  

    x_va_class1   = [np.array(x_va_class1[j])/CMOS_MAX for j in range(0,len(x_va_class1))]
    x_va_class2   = [np.array(x_va_class2[j])/CMOS_MAX for j in range(0,len(x_va_class2))]

    n_va_class1   = len(x_va_class1)
    n_va_class2   = len(x_va_class2)

    n_validation_examples = n_va_class1 + n_va_class2
    x_va = []
    t_va = []
    for i in  range(0,n_validation_examples):
        if i < n_va_class1:
            x_va.append(x_va_class1[i])
            t_va.append(0)
        else:
            x_va.append(x_va_class2[i-n_va_class1])
            t_va.append(1)

    x_va         = np.array(x_va)
    t_va         = np.array(t_va)

    torch_x_va      = torch.Tensor(x_va).float()
    torch_t_va      = torch.Tensor(t_va).type(torch.LongTensor)

    valid_xt      = TensorDataset(torch_x_va,torch_t_va)
    valid_loader  = DataLoader(valid_xt, batch_size = batch_size , shuffle=True)
    #make validation dataset , dataloader.

    #Create a separate make data function.

    print('')
    print('Beginning training for ' + str(n_epochs) + ' epochs , ' + 'with a batch size of ' + \
        str(batch_size) + ' and a training set of ' + str(int(n_ex_class1 + n_ex_class2)) + ' examples.')
    print('')

    training_losses       = []
    training_accuracies   = []
    validation_losses     = []
    validation_accuracies = []
    validation_rate_0s    = []
    validation_rate_1s    = []

    ctr = 0
    for epoch in tqdm(range(n_epochs),desc='Training progress'):
                
        if epoch == optimizer_update_epoch:
            print('Finetuning for ' + str(n_epochs - epoch) + ' more epochs with s.g.d at a l.r of ' + str(fine_tuning_lr))
            for g in optimizer.param_groups:
                g['lr'] = fine_tuning_lr

        losses_tr = []
        accs_tr   = []
        for j,data in enumerate(trainloader,0):

            f_composite.train()

            inputs,truths = data
            
            inputs = inputs.float()
            truths = torch.squeeze(truths)
            truths = truths.type(torch.LongTensor)

            if device == 'cuda':
                inputs = inputs.cuda()
                truths = truths.cuda()

            loss , ops = train_step(f_composite,inputs,truths,criterion,optimizer)
            
            training_preds = normalizer(ops)
            _,class_preds  = torch.max(training_preds.data,1)
            training_acc   = ((class_preds==truths).sum().item()/batch_size)*100

            losses_tr.append(round(loss.item(),4))
            accs_tr.append(training_acc)
        
        training_losses.append(np.mean(losses_tr))
        training_accuracies.append(np.mean(accs_tr))

        if validate:

            valid_losses     = []
            valid_accuracies = []
            valid_rate_0s    = []
            valid_rate_1s    = []
            for j,data in enumerate(valid_loader,0):

                inputs,truths = data
                nputs = inputs.float()
                truths = torch.squeeze(truths)
                truths = truths.type(torch.LongTensor)

                if device == 'cuda':
                    inputs = inputs.cuda()
                    truths = truths.cuda()

                [eval_acc , eval_loss , eval_rate_0 , eval_rate_1] = eval_step(f_composite,inputs,truths,normalizer,criterion)
                valid_losses.append(eval_loss)
                valid_accuracies.append(eval_acc)
                valid_rate_0s.append(eval_rate_0)
                valid_rate_1s.append(eval_rate_1)

            validation_losses.append(np.mean(valid_losses))
            validation_accuracies.append(np.mean(valid_accuracies))
            validation_rate_0s.append(np.mean(valid_rate_0s))
            validation_rate_1s.append(np.mean(valid_rate_1s))
    

    plt.figure()
    plt.title('Accuracy evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,100])
    plt.plot(training_accuracies,'r-')
    plt.plot(validation_accuracies,'g-')
    plt.plot(validation_rate_0s,'b-')
    plt.plot(validation_rate_1s,'y-')
    plt.legend(['training','validation accuracy' , 'hit rate - class 0' , 'hit rate - class 1'])
    plt.show()

    plt.figure()
    plt.title('Loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(training_losses,'r-')
    plt.plot(validation_losses,'g-')
    plt.legend(['training','validation'])
    plt.show()

    #wrap this around wandb -> Get the average accuracies and hit rates as well.