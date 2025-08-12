"""
GRU Model definition for intrusion detection
"""
import torch
import torch.nn as nn


class OptimizedGRUModel(nn.Module):
    """Optimized GRU model for intrusion detection"""
    
    def __init__(self, input_size, num_classes, hidden_size=64, dropout=0.2):
        super(OptimizedGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size, bias=False)
        
        # GRU layer
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bias=False
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_classes, bias=False)
    
    def forward(self, x):
        """Forward pass through the model"""
        # Ensure input has sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, features)
        
        # Project input to hidden size
        x = self.input_projection(x)
        
        # Pass through GRU
        gru_out, _ = self.gru(x)
        
        # Take the last output and apply dropout
        x = self.dropout(gru_out[:, -1, :])
        
        # Classification
        return self.classifier(x)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "OptimizedGRUModel",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


def create_model(config):
    """Factory function to create model with configuration"""
    return OptimizedGRUModel(
        input_size=config.num_features,
        num_classes=config.num_classes,
        hidden_size=config.hidden_size,
        dropout=config.dropout
    )


def save_model_info(model, output_path):
    """Save model information to file"""
    import json
    
    model_info = model.get_model_info()
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"[MODEL INFO] Saved model information to {output_path}")
    return model_info
