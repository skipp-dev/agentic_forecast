from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from pydantic import BaseModel
import os
import logging

logger = logging.getLogger(__name__)

class GraphTrainingData(BaseModel):
    symbols: List[str]
    times: List[datetime]
    features: Any  # np.ndarray [time, symbol, feature] - using Any to avoid pydantic validation issues with numpy
    adjacency: Any # np.ndarray [symbol, symbol] or edge list
    
    class Config:
        arbitrary_types_allowed = True

class GraphInferenceData(BaseModel):
    symbols: List[str]
    times: List[datetime]
    features: Any # np.ndarray
    adjacency: Any # np.ndarray
    
    class Config:
        arbitrary_types_allowed = True

class GraphForecastResult(BaseModel):
    predicted_return: Dict[str, Dict[int, float]]  # symbol -> horizon -> return
    embeddings: Optional[Dict[str, List[float]]] = None

import torch
from src.agents.graph_models.stgcnn_model import STGCNN

class GraphModelAgent:
    """
    Agent responsible for training and inference of Graph Neural Network models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, data: GraphTrainingData) -> None:
        """
        Train the graph model.
        """
        # Stub implementation
        num_nodes = len(data.symbols)
        # Assuming features shape [time, nodes, features]
        num_features = data.features.shape[2]
        num_timesteps_input = 10 # Configurable
        num_timesteps_output = 5 # Configurable
        
        self.model = STGCNN(num_nodes, num_features, num_timesteps_input, num_timesteps_output).to(self.device)
        self.model.train()
        logger.info("Training STGCNN model (stub)...")
        
    def predict(self, data: GraphInferenceData) -> GraphForecastResult:
        """
        Generate forecasts using the graph model.
        """
        if self.model is None:
            # Initialize dummy model if not trained (for testing/stub purposes)
            num_nodes = len(data.symbols)
            num_features = data.features.shape[2] if len(data.features.shape) > 2 else 1
            self.model = STGCNN(num_nodes, num_features, 10, 5).to(self.device)
            self.model.eval()
            
        self.model.eval()
        with torch.no_grad():
            # Prepare input tensor [batch, features, time, nodes]
            # Stub: just use random or reshape data.features
            # Assuming data.features is [time, nodes, features]
            # We need [batch, features, time, nodes]
            # Taking last 10 timesteps
            x = torch.tensor(data.features[-10:], dtype=torch.float32).to(self.device) # [time, nodes, features]
            x = x.permute(2, 0, 1).unsqueeze(0) # [1, features, time, nodes]
            
            adj = torch.tensor(data.adjacency, dtype=torch.float32).to(self.device)
            
            output = self.model(x, adj) # [1, nodes, horizon]
            output = output.cpu().numpy()[0] # [nodes, horizon]
            
        results: Dict[str, Dict[int, float]] = {}
        for i, symbol in enumerate(data.symbols):
            results[symbol] = {}
            for h in range(output.shape[1]):
                results[symbol][h+1] = float(output[i, h])
                
        return GraphForecastResult(predicted_return=results)
        
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        """
        if self.model:
            torch.save(self.model.state_dict(), path)
            
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        """
        # We need to know dimensions to init model before loading state dict
        # For stub, we'll assume defaults or need to save config with model
        pass
