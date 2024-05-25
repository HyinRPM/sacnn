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