# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:57:20 2024

@author: Adeka
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self):
        self.num_nodes = 307
        self.num_features = 3
        self.seq_length = 12
        self.hidden_dim = 64
        self.prediction_length = 3

class ASTGCN(nn.Module):
    def __init__(self, config):
        super(ASTGCN, self).__init__()
        self.config = config
        
        # Input projection
        self.input_fc = nn.Linear(config.num_features, config.hidden_dim)
        
        # Spatial attention
        self.spatial_fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.spatial_fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # GCN layers
        self.gc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.gc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Output projection - matches checkpoint dimensions
        # prediction_length * num_features = 3 * 3 = 9
        self.output_fc = nn.Linear(config.hidden_dim, 
                                 config.prediction_length * config.num_features)
    
    def forward(self, x, adj_matrix):
        """
        Forward pass of the ASTGCN model
        Args:
            x: Input tensor of shape [batch_size, seq_length, num_nodes, num_features]
            adj_matrix: Adjacency matrix of shape [num_nodes, num_nodes]
        """
        batch_size, seq_length, num_nodes, _ = x.size()
        
        # Project input features [batch * seq * nodes, features] -> [batch * seq * nodes, hidden]
        x_reshaped = x.contiguous().view(-1, self.config.num_features)
        x = self.input_fc(x_reshaped)
        x = x.view(batch_size, seq_length, num_nodes, self.config.hidden_dim)
        
        # Spatial attention
        spatial_hidden = self.spatial_fc1(x)
        spatial_scores = self.spatial_fc2(torch.tanh(spatial_hidden))
        spatial_attention = F.softmax(spatial_scores, dim=2)
        x = x * spatial_attention
        
        # Temporal attention
        # Reshape: [batch, seq, nodes, hidden] -> [batch * nodes, seq, hidden]
        x_temporal = x.permute(0, 2, 1, 3).contiguous()
        x_temporal = x_temporal.view(-1, seq_length, self.config.hidden_dim)
        
        # Apply temporal attention
        x_temporal, _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
        
        # Reshape back: [batch * nodes, seq, hidden] -> [batch, seq, nodes, hidden]
        x = x_temporal.view(batch_size, num_nodes, seq_length, self.config.hidden_dim)
        x = x.permute(0, 2, 1, 3)
        
        # Graph convolution
        # Process each time step
        outputs = []
        for t in range(seq_length):
            h = x[:, t]  # [batch, nodes, hidden]
            h = torch.matmul(adj_matrix, h)  # Graph conv
            h = self.gc1(h)
            h = F.relu(h)
            h = torch.matmul(adj_matrix, h)  # Second graph conv
            h = self.gc2(h)
            h = F.relu(h)
            
            # Project to output features
            out = self.output_fc(h)  # [batch, nodes, prediction_length * features]
            # Reshape to separate prediction length and features
            out = out.view(batch_size, num_nodes, self.config.prediction_length, self.config.num_features)
            outputs.append(out)
        
        # Stack time steps
        outputs = torch.stack(outputs, dim=1)  # [batch, seq, nodes, pred_length, features]
        
        # Return predictions for last time step and all prediction lengths
        return outputs[:, -1]  # [batch, nodes, pred_length, features]

class PredictionRequest(BaseModel):
    sequence: list
    adj_matrix: list

app = FastAPI(title="Traffic Prediction API")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = ModelConfig()
model = ASTGCN(config).to(device)

try:
    checkpoint = torch.load('production_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

model.eval()

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input to tensors
        sequence = torch.FloatTensor(request.sequence).to(device)
        adj_matrix = torch.FloatTensor(request.adj_matrix).to(device)
        
        # Log input shapes
        logger.info(f"Input sequence shape: {sequence.shape}")
        logger.info(f"Input adj_matrix shape: {adj_matrix.shape}")
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence, adj_matrix)
            logger.info(f"Output prediction shape: {prediction.shape}")
            
        return {
            "prediction": prediction.cpu().numpy().tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "version": "1.0",
        "metrics": checkpoint.get('performance_metrics', {}),
        "last_updated": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)