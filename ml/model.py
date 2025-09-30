import torch
import torch.nn as nn
import math
from typing import Optional
from dataclasses import dataclass

from bb_toolkit import BasicBlockTokenizer
from ml.embedding import DynamicEmbedding

@dataclass
class ModelConfig:
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    dropout: float

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Embeddings of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class OnlineBasicBlockPredictor(nn.Module):
    def __init__(
        self,
        device: torch.device,
        initial_vocab_size: int = 1000,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        self.config = config or ModelConfig(
            context_length=64,
            embedding_dim=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
        )

        self.embedding = DynamicEmbedding(
            embedding_dim=self.config.embedding_dim,
            initial_vocab_size=initial_vocab_size,
            padding_idx=BasicBlockTokenizer.PADDING_TOKEN,
            device=device
        )
        
        self.pos_encoding = PositionalEncoding(self.config.embedding_dim, max_len=self.config.context_length * 2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.config.num_layers)
        
        self.output_projection = None
        self.dropout = nn.Dropout(self.config.dropout)

    def _ensure_output_projection(self, vocab_size: int):
        """Ensure output projection matches current vocabulary size"""
        if self.output_projection is None or self.output_projection.out_features < vocab_size:
            old_proj = self.output_projection
            self.output_projection = nn.Linear(self.config.embedding_dim, vocab_size).to(self.device)
            
            # copy existing weights if we had a previous projection
            if old_proj is not None:
                with torch.no_grad():
                    old_size = old_proj.out_features
                    self.output_projection.weight[:old_size] = old_proj.weight.to(self.device)
                    self.output_projection.bias[:old_size] = old_proj.bias.to(self.device)
                    
                    # init new weights
                    nn.init.normal_(self.output_projection.weight[old_size:], std=0.02)
                    nn.init.zeros_(self.output_projection.bias[old_size:])
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through the model.

        Args:
            input_ids: Token sequences [batch_size, sequence_length]
            labels: Target tokens for training [batch_size, sequence_length]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        embeddings = self.pos_encoding(embeddings)
        
        # create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        
        # transform through encoder with causal mask
        transformed = self.transformer(embeddings, mask=mask)

        current_vocab_size = max(self.embedding.get_vocab_size(), input_ids.max().item() + 1)
        if labels is not None:
            current_vocab_size = max(current_vocab_size, labels.max().item() + 1)
        
        self._ensure_output_projection(current_vocab_size)

        logits = self.output_projection(transformed)
        
        result = {'logits': logits}
        
        # compute loss
        if labels is not None:
            # shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result['loss'] = loss
        
        return result

    def predict_next_block(self, input_sequence: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Predict the next block given an input sequence.

        Args:
            input_sequence: Token sequence [sequence_length]
            top_k: Number of top predictions to return

        Returns:
            List of (token_id, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            
            if len(input_sequence) > self.config.context_length:
                input_sequence = input_sequence[-self.config.context_length:]
            
            # add batch dimension
            input_ids = input_sequence.unsqueeze(0)
            
            outputs = self.forward(input_ids)

            # scores for the last token in the sequence
            logits = outputs['logits'][0, -1, :]

            # normalizes probabilities, since logits can be any real number
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    @property
    def device(self) -> torch.device:
        return self.embedding.device
    
    def get_vocab_size(self) -> int:
        return self.embedding.get_vocab_size()