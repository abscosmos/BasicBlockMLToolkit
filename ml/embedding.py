import torch
import torch.nn as nn

class ShardedEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        shard_size: int = 2048,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embedding_dim: int = embedding_dim
        self.shard_size: int = shard_size
        self.num_active_tokens: int = 0

        self.shards = nn.ModuleList()


    # fixme
    def _reserve(self, new_capacity: int):
        while new_capacity >= self.capacity:
            self._create_shard()

    def _create_shard(self) -> None:
        new_shard = nn.Embedding(self.shard_size, self.embedding_dim)
        # random init
        nn.init.normal(new_shard.weight, mean=0, std=0.1)
        self.shards.append(new_shard)

    def _get_shard_idx_and_offset(self, token_id: int) -> tuple[int, int]:
        return token_id // self.shard_size, token_id % self.shard_size

    @property
    def capacity(self) -> int:
        return len(self.shards) * self.shard_size

    def average_embedding(self) -> torch.Tensor:
        """
        :return: average embedding vector in the embedding space
        """
        if self.num_active_tokens == 0:
            return torch.zeros(self.embedding_dim)

        if self.num_active_tokens > self.capacity:
            raise ValueError("Number of active tokens exceeds capacity.")

        embs: list[torch.Tensor] = []
        for token_id in range(self.num_active_tokens):
            shard_idx, offset = self._get_shard_idx_and_offset(token_id)
            embs.append(self.shards[shard_idx].weight.data[offset])

        # combines all the tensors into a 2d tensor
        stacked = torch.stack(embs)
        # mean across each of the tensors in the 2d tensor
        return torch.mean(stacked, dim=0)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Computes token embeddings for a batch of token IDs
        :param token_ids: input tokens in shape (batch_size, seq_len)
        :return: embeddings of tokens in shape (batch_size, seq_len, embedding_dim)
        """

        max_token_id: int = token_ids.max().item()
        if max_token_id >= self.num_active_tokens:
            self._reserve(max_token_id)

        batch_size, seq_len = token_ids.shape
        out_embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim, device=token_ids.device)

        for shard_idx, shard in enumerate(self.shards):
            # get start and end token ids for this shard
            start_id = shard_idx * self.shard_size
            end_id = start_id + self.shard_size

            # selects tokens that belong to this shard
            mask: torch.BoolTensor = (token_ids >= start_id) & (token_ids < end_id)

            if mask.any():
                # offsets within the shard
                # extracts all elements where the mask is true (token belongs to the current shard)
                # then maps to offsets within shard
                offsets = (token_ids[mask] - start_id).long()
                # retrieves the embeddings at each of the offsets, and assigns it to the
                # correct position in the out tensor
                out_embeddings[mask] = shard(offsets)

        return out_embeddings