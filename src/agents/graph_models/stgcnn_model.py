import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Neural Network (STGCNN) stub.
    Inspired by SP100 GNN repo.
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(STGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        
        # Placeholder layers
        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=(1, 1))
        self.fc = nn.Linear(64 * num_timesteps_input, num_timesteps_output)
        
    def forward(self, x, A):
        """
        x: [batch, num_features, num_timesteps, num_nodes]
        A: [num_nodes, num_nodes] adjacency matrix
        """
        # Simple stub implementation
        # In a real STGCNN, we would use graph convolutions with A
        
        batch_size = x.size(0)
        
        # Dummy convolution
        out = self.conv1(x) # [batch, 64, time, nodes]
        
        # Flatten
        out = out.permute(0, 3, 1, 2).contiguous() # [batch, nodes, 64, time]
        out = out.view(batch_size, self.num_nodes, -1) # [batch, nodes, 64*time]
        
        # Predict
        out = self.fc(out) # [batch, nodes, horizon]
        
        return out
