from __future__ import print_function

#from agents.imitation.frn import FRN   xxx
from .fusion import REFusion 
import numpy as np

import torch

import torch.nn.functional as F

from .model import ResNet_fpn_fusion
from .utils import Bottleneck


def weight_ones(shape, name):
    initial = torch.Tensor(1.0, shape=shape, name=name) #torch.ones (shape) ?
    return torch.nn.Parameter(initial)


#def weight_xavi_init(shape, name):
    # potential problem
    #initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    #return initial


def bias_variable(shape, name):
    initial = torch.Tensor(0.1, shape=shape, name=name)
    return torch.nn.Parameter(initial)


class Mymodel(torch.nn.Module): #geändert

    def __init__(self, image_shape, Device, use_fusion=False):#bol übergeben mit frn, events
        super().__init__()#geändert
        self.use_fusion = use_fusion #if condition reinmachen 30.04
        
        
        
        
# start pytorch

        self._count_activations = 0
        self._count_fc = 0
        

        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        self._dout = torch.Tensor(self.dropout_vec)

        self._image_shape = image_shape
        IN = image_shape[-1]
        self._conv_kernels = [5, 3, 3, 3, 3, 3, 3, 3]
        self._conv_stride = [2, 1, 2, 1, 2, 1, 1, 1]
        self._conv_padding = [0, 0, 0, 0, 0, 0, 0, 0]
        self._conv_channel_in = [IN, 32, 32, 64, 64, 128, 128, 256, 256]
        self._conv_channel_out = [32, 32, 64, 64, 128, 128, 256, 256]
        self.convs = torch.nn.ModuleList()

        #Fc-layer anlegen
        if self.use_fusion:
            first_fc_in = 7680     
            self.frn = ResNet_fpn_fusion(
            dataset_name="ddd17",
            event_block=Bottleneck,
            rgb_block=Bottleneck,
            event_layers=[3, 4, 6, 3],
            rgb_layers=[3, 4, 6, 3]
        ).to(Device)

        else:
            first_fc_in = 200192    
            for i in range(0, 8):
                self.convs.append(torch.nn.Conv2d(
                    in_channels=self._conv_channel_in[i], 
                    out_channels=self._conv_channel_out[i], 
                    kernel_size=self._conv_kernels[i], 
                    stride=self._conv_stride[i], 
                    padding=0, 
                    device=Device))
                
        self._fc_layer_sizes_in = [first_fc_in, 512, 1, 128, 512, 256, 256]

        self._fc_layer_sizes_out = [512, 512, 128, 128, 256, 256, 1]

       

            
        
        self.fcs = torch.nn.ModuleList()# geändert
        #self.fcs.append(torch.nn.Linear(200192, 512).to(Device))
        self.fc_joint = torch.nn.Linear(512+128, 512).to(Device)
        for i in range(0, len(self._fc_layer_sizes_in)):
            self.fcs.append(
                torch.nn.Linear(
                    self._fc_layer_sizes_in[i],
                    self._fc_layer_sizes_out[i]
                ).to(Device)
            )
        #print(self.fcs)


        # Dropouts
        self.douts = torch.nn.ModuleList()
        for i in range(len(self.fcs) + len(self.convs)):
            drop_prob = 1.0 - self.dropout_vec[i]
            self.douts.append(torch.nn.Dropout(drop_prob))

        #Batchnorm Layer
        self.bns = torch.nn.ModuleList()
        for i in range(0,8):
            self.bns.append(
                torch.nn.BatchNorm2d(self._conv_channel_out[i]).to(Device)# geändert
            )    
        #Activation
        self.relu = torch.nn.ReLU()
        if self.use_fusion:
            print("FRN branch active")
        else:
            print("CIL branch active")
        print("init_fertig")

        

    def activation(self,x):
        self._count_activations +=1
        return self.relu(x)
        
    #def fc(self, x, output_size):
        #self._count_fc += 1

        #filters_in = x.shape[-1]
        #linear = torch.nn.Linear(filters_in,output_size).to(x.device)
        #torch.nn.init.xavier_uniform_(linear.weight)
        ##torch.nn.init.constant_(linear.bias,0.1)
        #return linear(x)
    
    def conv_block(self, x, i):
        #print(" === Conv", i,)#geändert
        #print("conv index:", i)
        #print("expected in_channels:", self.convs[i].in_channels)
        #print("actual input shape:", x.shape)
        x = self.convs[i](x)
        x = self.bns[i](x)
        x = self.douts[i](x)
        return self.activation(x)
    
    
    
    def fc_block(self, x, i):
        #print(" === FC", i, "  :  ", self._fc_layer_sizes_in[i], "->", self._fc_layer_sizes_out[i])

        x = self.fcs[i](x)
        x = self.douts[i](x)#geändert

       
        return self.activation(x)
    
        


    def forward(self, input_image, input_data, input_event=None): 
    #def forward(self, input_image, input_data,): 
        #def forward(self, input_image, input_event, input_data,):   xxx
        #branches = []
        #forward funktion frn aufrufen mit if else

        
        if self.use_fusion:
            print("FRN aktiv")
            x = self.frn((input_image, input_event))  
            #print("FRN requires_grad:", x.requires_grad)
            #print("FRN mean/std:", x.mean().item(), x.std().item())
            x = x[-1]  
            x = torch.flatten(x, 1)
            print("FRN output type:", type(x))
        else:
            print("CIL aktiv")
            x = input_image
            for i in range(8):
                x = self.conv_block(x, i)
            x = torch.flatten(x, 1)
            print("CIL flatten shape:", x.shape)

            print("flatten requires_grad:", x.requires_grad)
            print("flatten mean/std:", x.mean().item(), x.std().item())
        #if self.use_fusion:
            #x_rgb = input_image
            #x_evt = input_event
            #for i in range(8):
                #x_rgb = self.conv_block_rgb(x_rgb, i)
                #x_evt = self.conv_block_evt(x_evt, i)
            #x = self.refusion(x_rgb, x_evt)
            #x = self.post_fusion(x)
        #else:
            #x = input_image
            #for i in range(8):
                #x = self.conv_block(x, i)
    
        """ reshape """
        #x = torch.flatten(x, 1)
        #print(x.shape)#dimension davon (-1) ist input von fc in_features

        #CIL-Branch

        for i in range(2):
            x = self.fc_block(x, i)

        print("after fc0 requires_grad:", x.requires_grad)
        print("after fc0 mean/std:", x.mean().item(), x.std().item())
        """Process Control"""

        """ Speed (measurements)"""
        #speed = torch.tensor(input_data).float().to(x.device)
        #speed = speed.view(1, -1)
        ##print("Speed shape:", speed.shape)
        #speed = self.fc_block(speed, 2)
        #speed = self.fc_block(speed, 3)

        speed = input_data.float().to(x.device)
        speed = self.fc_block(speed, 2) # nur noch speed reintun!
        speed = self.fc_block(speed, 3)

        """ Joint sensory """
        j = torch.cat([x, speed], dim=1)
        j = self.fc_joint(j)
        j = self.relu(j)

        """Start BRANCHING"""
        branch_output = self.fc_block(j, 4)
        branch_output = self.fc_block(branch_output, 5)
        branch_output = self.fcs[6](branch_output) 


        #print(branch_output, torch.squeeze(branch_output, 0))
        return branch_output

# end pytorch









