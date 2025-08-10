import torch
import torch.nn as nn

from bb_toolkit import BasicBlockTokenizer
from embedding import ShardedEmbedding

class BasicBlockPredictor(nn.Module):
    def __init__(
            self,
            embedding: ShardedEmbedding,
            tokenizer: BasicBlockTokenizer,
            # number of attention heads
            n_head: int = 4,
            # number of transformer encoder layers
            num_layers: int = 2,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embedding: ShardedEmbedding = embedding
        self.tokenizer: BasicBlockTokenizer = tokenizer

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=n_head
            ),
            num_layers=num_layers
        )

        self.output_linear = nn.Linear(self.embedding_dim, len(tokenizer))

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if self.num_active_tokens == 0:
            return torch.zeros(self.embedding_dim)

        embeddings: torch.Tensor = self.embedding(token_ids)
        embeddings = embeddings.transpose(0, 1)

        transformer_out: torch.Tensor = self.transformer(embeddings)
        transformer_out = transformer_out.transpose(0, 1)

        logits = self.output_linear(transformer_out)

        return logits

    def predict_next_block(self, input_sequence: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        raise NotImplementedError()


    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def to(self, device: torch.device) -> None:
        self.transformer.to(device)
        self.embedding.to(device)

def create_model(vocab_size: int, context_length: int = 64) -> BasicBlockPredictor:
    raise NotImplementedError()