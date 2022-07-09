# Importing all the essential libraries

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# # Constructing the model

# class neural_network(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(neural_network, self).__init__()
#         self.fc1 = nn.Linear(in_features=input_size, out_features=50)
#         self.fc2 = nn.Linear(in_features=50, out_features=num_classes)
        
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x


# # Train the network

# for epoch in range(epochs):
#     for batch, (data, target) in enumerate(train_loader):
#         # Obtaining the cuda parameters
#         data = data.to(device=device)
#         target = target.to(device=device)
        
#         # Reshaping to suit our model
#         data = data.reshape(data.shape[0], -1)
        
#         # Forward propogation
#         score = model(data)
#         loss = criterion(score, target)
        
#         # Backward propagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()