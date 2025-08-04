import torch
import torch.nn as nn
from typing import Optional
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class BasicBlockPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=context_length,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )

        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> CausalLMOutputWithCrossAttentions:
        """
        Forward pass through the model.

        Args:
            input_ids: Token sequences [batch_size, sequence_length]
            labels: Target tokens for training [batch_size, sequence_length]

        Returns:
            Model outputs with loss (if labels provided) and logits
        """
        return self.transformer(input_ids=input_ids, labels=labels)

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
        # speeds up inference, no gradient tracking needed since not training
        with torch.no_grad():
            # model expects inputs in batches, [1, sequence_length]
            input_ids = input_sequence.unsqueeze(0)

            # makes the predictions
            outputs = self.transformer(input_ids)
            # scores for the last token in the sequence
            logits = outputs.logits[0, -1, :]

            # normalizes probabilities, since logits can be any real number
            probs = torch.softmax(logits, dim=-1)

            # finds the top probabilities
            top_probs, top_indices = torch.topk(probs, top_k)

            return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    @property
    def device(self) -> torch.device:
        return self.transformer.device

def create_model(vocab_size: int, context_length: int = 64) -> BasicBlockPredictor:
    if vocab_size < 1000:
        embedding_dim = 128
        num_layers = 4
        num_heads = 4
    elif vocab_size < 5000:
        embedding_dim = 256
        num_layers = 6
        num_heads = 8
    else:
        embedding_dim = 512
        num_layers = 8
        num_heads = 8

    model = BasicBlockPredictor(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1
    )

    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Context length: {context_length}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of layers: {num_layers}")

    return model