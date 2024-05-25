import torch
import torch.nn as nn
import numpy as np

class self_attention(nn.Module):
    def __init__(self,in_dim,q_dim,k_dim,v_dim,v_mapping):
        super(self_attention,self).__init__()

        self.chanel_in = in_dim
        self.v_mapping = v_mapping

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = q_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = k_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = v_dim , kernel_size= 1)

        self.o_proj = nn.Conv2d(in_channels = v_dim , out_channels = in_dim , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
        
    def forward(self,x):
        m_batchsize,C,width,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)   

        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)

        if self.v_mapping == True:
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
        else:
            proj_value = x.view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        
        if self.v_mapping == True:
            out = out.view(m_batchsize,self.v_dim,width,height)
            out = self.o_proj(out)
        else:
            out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        
        return out,attention

class net_one_neuron_ff_sa_cnn_star_incr_fc2(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_cpbs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3)
         )
        
        self.attention = self_attention(in_dim=30,q_dim=5,k_dim=5,v_dim=5,v_mapping=True)

        self.layers_sa_reg = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3)
        )

        self.b_cpbs = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),

            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(9 * 9 * 30, 1)

        for param in self.a_cpbs.parameters():
            param.requires_grad = False
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.layers_sa_reg.parameters():
            param.requires_grad = False
        for param in self.b_cpbs.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.a_cpbs(x)
        x,_ = self.attention(x) 
        x = self.layers_sa_reg(x)
        x = self.b_cpbs(x)
        x = self.flatten(x)
        x = self.fcl(x)
        return x


class seperate_core_model_ff_sa_cnn_star_incr_fc2(nn.Module):
    def __init__(self,num_neurons):
        super().__init__()
        self.models = nn.ModuleList([net_one_neuron_ff_sa_cnn_star_incr_fc2() for i in range(num_neurons)])
        self.num_neurons = num_neurons

    def forward(self, x):
        outputs = [self.models[i].forward(x) for i in range(self.num_neurons)]
        outputs = torch.stack(outputs, dim=1)
        return outputs.reshape((outputs.shape[0], outputs.shape[1]))

def model_ff_sa_cnn_star_incr_fc2(num_neurons):
    return seperate_core_model_ff_sa_cnn_star_incr_fc2(num_neurons=num_neurons)

def model_ff_sa_cnn_star_incr_fc2_one_neuron():
	 return net_one_neuron_ff_sa_cnn_star_incr_fc2()