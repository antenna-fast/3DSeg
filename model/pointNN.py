"""
Author: ANTenna on 2022/1/13 3:48 下午
aliuyaohua@gmail.com

Description:
The most Naive Model for 3D semantic segmentation
Shared MLP
"""

import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Define model elements here
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Shared MLP for point cloud learning
        self.shared_mlp = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU(),
                                        nn.Linear(self.hidden_dim, 128), nn.ReLU(),
                                        nn.Linear(128, 256), nn.ReLU(),
                                        nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, self.hidden_dim), nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.out_dim),
                                        )
    # Model Architecture
    def forward(self, x):
        assert x.shape[-1] == self.in_dim, 'ERROR Input Data Shape:{}, Expected feature dimension:{} .. '.\
            format(x.shape, self.in_dim)
        
        x = self.shared_mlp(x)
        return x


if __name__ == '__main__':
    test_data = torch.ones(2, 100, 3)
    test_label = torch.ones(2, 100)
    model = NN(3, 64, 10)
    output = model(test_data)
