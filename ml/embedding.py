import torch
import torch.nn as nn
from typing import Optional

class DynamicEmbedding(nn.Module):
    """
    High-performance dynamic embedding that can grow its vocabulary during training/inference.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        initial_vocab_size: int = 1000,
        growth_factor: float = 1.5,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.growth_factor = growth_factor
        self.padding_idx = padding_idx
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(
            num_embeddings=initial_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            device=self._device
        )

        self.current_vocab_size = 0
        self.total_expansions = 0
        
        self._initialize_embeddings(0, initial_vocab_size)
    
    def _initialize_embeddings(self, start_idx: int, end_idx: int):
        """Initialize embedding weights for tokens in range [start_idx, end_idx)"""
        with torch.no_grad():
            # small random initialization (similar to BERT/GPT)
            nn.init.normal_(
                self.embedding.weight[start_idx:end_idx], 
                mean=0.0, 
                std=0.02
            )

            if self.padding_idx is not None and start_idx <= self.padding_idx < end_idx:
                self.embedding.weight[self.padding_idx].fill_(0.0)
    
    def expand_vocabulary(self, target_vocab_size: int):
        """
        Expand the embedding table to accommodate target_vocab_size tokens.
        Uses growth_factor to pre-allocate extra capacity for efficiency.
        """
        if target_vocab_size <= self.embedding.num_embeddings:
            return
        
        new_size = max(
            int(target_vocab_size * self.growth_factor),
            target_vocab_size + 1000
        )
        
        old_embedding = self.embedding
        self.embedding = nn.Embedding(
            num_embeddings=new_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            device=self._device
        )

        with torch.no_grad():
            self.embedding.weight[:old_embedding.num_embeddings].copy_(old_embedding.weight)

            self._initialize_embeddings(
                old_embedding.num_embeddings, 
                new_size
            )
        
        self.total_expansions += 1
        
        del old_embedding
    
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Standard embedding lookup with automatic vocabulary expansion.
        
        Args:
            token_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            embeddings: Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        # bounds check
        if token_ids.numel() == 0:
            return torch.empty((*token_ids.shape, self.embedding_dim), device=self._device)

        max_token_id = token_ids.max().item()
        if max_token_id >= self.embedding.num_embeddings:
            self.expand_vocabulary(max_token_id + 1)
        
        self.current_vocab_size = max(self.current_vocab_size, max_token_id + 1)
        
        return self.embedding(token_ids)
    
    def get_vocab_size(self) -> int:
        """Return the current vocabulary size (highest token ID + 1)"""
        return self.current_vocab_size
    
    def get_capacity(self) -> int:
        """Return the current embedding table capacity"""
        return self.embedding.num_embeddings
    
    def get_utilization(self) -> float:
        """Return vocabulary utilization ratio (used/capacity)"""
        return self.current_vocab_size / self.embedding.num_embeddings if self.embedding.num_embeddings > 0 else 0.0
    
    @property
    def device(self) -> torch.device:
        # Return the actual device of the embedding weights
        return self.embedding.weight.device
    
    def to(self, device: torch.device):
        """Move embedding to specified device"""
        self._device = device
        self.embedding = self.embedding.to(device)
        return super().to(device)
    
    def state_dict(self, *args, **kwargs):
        """Custom state dict that includes metadata"""
        state = super().state_dict(*args, **kwargs)
        state['current_vocab_size'] = self.current_vocab_size
        state['total_expansions'] = self.total_expansions
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom state dict loading that restores metadata"""

        # metadata
        if 'current_vocab_size' in state_dict:
            self.current_vocab_size = state_dict.pop('current_vocab_size')
        else:
            self.current_vocab_size = 0
            
        if 'total_expansions' in state_dict:
            self.total_expansions = state_dict.pop('total_expansions')
        else:
            self.total_expansions = 0
        
        # weights
        return super().load_state_dict(state_dict, strict)