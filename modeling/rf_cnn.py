import torch
import torch.nn as nn
import numpy as np

class net_one_neuron_rf_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
        )
        self.Linear = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(-1,32,81)
        x = x[:,:,40]
        x = self.Linear(x)
        return x


class seperate_core_model_rf_cnn(nn.Module):
    def __init__(self,num_neurons):
        super().__init__()
        self.models = nn.ModuleList([net_one_neuron_rf_cnn() for i in range(num_neurons)])
        self.num_neurons = num_neurons

    def forward(self, x):
        outputs = [self.models[i].forward(x) for i in range(self.num_neurons)]
        outputs = torch.stack(outputs, dim=1)
        return outputs.reshape((outputs.shape[0], outputs.shape[1]))

def model_rf_cnn(num_neurons):
    return seperate_core_model_rf_cnn(num_neurons=num_neurons)

def model_rf_cnn_one_neuron():
    return net_one_neuron_rf_cnn()