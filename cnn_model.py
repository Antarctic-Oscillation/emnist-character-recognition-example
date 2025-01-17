import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Initialize shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        return F.relu(out)

class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int, initial_channels: int = 64, num_layers: int = 3, dropout_rate: float = 0.5) -> None:
        super().__init__()
        
        self.input_block = nn.Sequential(
            nn.Conv2d(1, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # Build residual layers
        self.layers = nn.ModuleList()
        current_channels = initial_channels
        for i in range(num_layers):
            out_channels = initial_channels * (2 ** i)
            layer = self.make_layer(
                ResidualBlock,
                current_channels,
                out_channels,
                num_blocks=2,
                stride=1
            )
            self.layers.append(layer)
            current_channels = out_channels
            
        # Output layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        fc_hidden_dim = min(128, current_channels // 2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        
    def make_layer(self, block: nn.Module, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        
        # Remaining blocks maintain channel dimensions
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_block(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        return self.classifier(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        x = self.input_block(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def full_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and embeddings."""
        x = self.input_block(x)
        for layer in self.layers:
            x = layer(x)
        embedding = x
        
        x = self.avg_pool(x)
        logits = self.classifier(x)
        return logits, embedding