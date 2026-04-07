import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class SmoothK(nn.Module):
    """
    Smooth K activations by subtracting per-channel mean to reduce outliers.
    
    Two strategies:
    1. 'per_token': Subtract mean for each token independently (along head_dim)
       - Reduces token-level outliers
       - Shape: [B, H, S, 1] - different mean for each token
       
    2. 'per_channel': Subtract global mean for each channel (along batch/seq)
       - Reduces channel-level outliers (some channels consistently larger)
       - Shape: [1, 1, 1, D] - same mean for all tokens in a channel
    """
    
    def __init__(self, enabled: bool = True, strategy: str = 'per_channel'):
        super().__init__()
        self.enabled = enabled
        self.strategy = strategy  # 'per_token' or 'per_channel'
    
    def forward(self, key_states: torch.Tensor) -> torch.Tensor:
        """        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            Smoothed key_states with mean subtracted
        """
        if not self.enabled:
            return key_states
        
        if self.strategy == 'per_token':
            # Subtract mean for each token (along head_dim dimension)
            # Shape: [B, H, S, 1]
            channel_mean = key_states.mean(dim=-1, keepdim=True)
            
        elif self.strategy == 'per_channel':
            # Subtract global mean for each channel (along batch, heads, seq dimensions)
            # Shape: [1, 1, 1, D]
            channel_mean = key_states.mean(dim=(0, 1, 2), keepdim=True)
            
        else:
            raise ValueError(f"Unknown smoothing strategy: {self.strategy}")
        
        key_states_smooth = key_states - channel_mean
        return key_states_smooth