from typing import Dict, List, Optional
from bb_toolkit import TraceData, SymbolizedBasicBlock

class BasicBlockTokenizer:
    def __init__(self):
        self.block_to_token: Dict[SymbolizedBasicBlock, int] = {}
        self.token_to_block: Dict[int, SymbolizedBasicBlock] = {}

        special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        for block, token in special_tokens.items():
            self.block_to_token[block] = token
            self.token_to_block[token] = block

        self.next_token_id: int = len(special_tokens)

    def add_block(self, symbolized_block: SymbolizedBasicBlock) -> int:
        """Add a symbolized basic block to vocabulary and return its token ID."""
        if symbolized_block not in self.block_to_token:
            token_id = self.next_token_id
            self.block_to_token[symbolized_block] = token_id
            self.token_to_block[token_id] = symbolized_block
            self.next_token_id += 1
            return token_id
        return self.block_to_token[symbolized_block]

    def get_token_id(self, symbolized_block: SymbolizedBasicBlock) -> int:
        # return <UNK> if block not found
        return self.block_to_token.get(symbolized_block, 1)

    def get_block(self, token_id: int) -> Optional[SymbolizedBasicBlock]:
        """Get symbolized basic block for a token ID, None if not found."""
        return self.token_to_block.get(token_id)

    def process_trace(self, trace_data: TraceData) -> List[int]:
        token_sequence = []

        for block_location in trace_data.order:
            symbolized_block = trace_data.blocks[block_location].symbolize()

            token_id = self.add_block(symbolized_block)
            token_sequence.append(token_id)

        return token_sequence

    def __len__(self) -> int:
        return len(self.block_to_token)