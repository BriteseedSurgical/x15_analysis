import torch
import torch.nn as nn
import torch.nn.functional as f
from   torch.nn.utils import weight_norm as wn

import numpy as np
from   torchsummary import summary

"""
Functions : forward() , backward() , initialize() , load() , save() , get_layer() , set_layer()
"""

minus_infinity = -1 * 1e10 

#Input : 1 * 4 * 99
class x4_def_conv_block(nn.Module):
    
    def __init__(self , n_channels_in , nf , conv_kernel_size  = (4,4)):
        super(x4_def_conv_block,self).__init__()

        self.pad  = nn.ConstantPad2d((int(conv_kernel_size[1]/2),int((conv_kernel_size[1]/2)-1),0,0),value = 0)
        self.conv = nn.Conv2d(in_channels = n_channels_in , out_channels = nf , kernel_size = conv_kernel_size)
        self.bn   = nn.BatchNorm2d(nf)
        self.act  = nn.ReLU()
        #BatchNorm : (?) Does not feed into other convolutions , sparse feature map.
        #Pooling   : (?) Preserve spatial resolution for now.
        #Padding same in X and Y ? -> Not intuitive for the data I know.

    def forward(self,x):
        
        y = self.pad(x)
        y = self.conv(y)
        y = self.bn(y)
        y = self.act(y)

        return y

class x4_def_fc_block(nn.Module):

    #To calculate : n_in
    def __init__(self , n_in = -1, layer_widths = [99 , 16 , 64] , dropout_fractions = [0.5,0.5,0] , dropout = False):
        super(x4_def_fc_block,self).__init__()

        self.n_layers = len(layer_widths)
        self.fc_stack = nn.ModuleList()
        for j in range(0,self.n_layers):
            if j == 0:
                self.fc_stack.append(nn.Linear(n_in , layer_widths[0]))
            else:
                self.fc_stack.append(nn.Linear(layer_widths[j-1] , layer_widths[j]))
            if j != self.n_layers - 1:
                self.fc_stack.append(nn.ReLU())
            if dropout:
                self.fc_stack.append(nn.Dropout(dropout_fractions[j]))
        self.n_layers = len(self.fc_stack)

    def forward(self,x):

        y = x
        for j in range(0,self.n_layers):
            y = self.fc_stack[j].forward(y)
        
        return y

class x4_def_pretrain(nn.Module):

    def __init__(self , ip_length = 99 ,  n_ip_channels = 1 , n_conv_filters = [64,16] , conv_kernel_sizes = [(4,4),(1,1)]):
        super(x4_def_pretrain,self).__init__()

        self.n_conv = len(n_conv_filters)
        self.layers = nn.ModuleList()

        for j in range(0,self.n_conv):
            if j == 0:
                self.layers.append(x4_def_conv_block(n_ip_channels , n_conv_filters[j] , conv_kernel_sizes[j]))
            else:
                self.layers.append(x4_def_conv_block(n_conv_filters[j-1] , n_conv_filters[j] , conv_kernel_sizes[j]))
        self.layers.append(nn.Flatten())
        self.layers.append(x4_def_fc_block(n_in = 1584 , layer_widths = [1024,512]))

        self.n_layers = len(self.layers)

        return

    def forward(self,x):

        y = x
        for j in range(0,self.n_layers):
            y = self.layers[j].forward(y)
            
        return y

class fc_filler(nn.Module):

    def __init__(self , ip_length = 128 , fc_lengths = [2]):
        super(fc_filler,self).__init__()

        self.n_fc = len(fc_lengths)
        self.layers = nn.ModuleList()

        for j in range(0,self.n_fc):
            if j == self.n_fc - 1:
                _ = 0
            else:
                self.layers.append(nn.Dropout(0.5))
    
            if j == 0:
                self.layers.append(nn.Linear(ip_length , fc_lengths[j]))
            else:
                self.layers.append(nn.Linear(fc_lengths[j-1] , fc_lengths[j]))

            if j == self.n_fc-1:
                _ = 0
            else:
                self.layers.append(nn.ReLU())
                

        self.n_layers = len(self.layers)

    def forward(self,x):

        y = x
        for j in range(0,self.n_layers):
            y = self.layers[j].forward(y)
        
        return y
            


#Make it a composite model ? But then , you cannot use the built in 
#Cosine embedding loss.
"""
class x4_pretrain_wrapper(nn.Module):
    def __init__(self , x4_pretrain):
        super(x4_pretrain_wrapper,self).__init__()
"""
#Architecture 
#Spatio-Spectral filtering 1 (retain size) -> Spectral features (n 1*1 spatial conv) -> Flatten -> FC head (feature mixing)
#Large number of initial filters