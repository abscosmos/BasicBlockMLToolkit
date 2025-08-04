import torch
from bb_toolkit import BasicBlockTokenizer, SymbolizedBasicBlock
from ml.model import BasicBlockPredictor

def make_inference(
    model: BasicBlockPredictor,
    tokenizer: BasicBlockTokenizer,

    input: list[SymbolizedBasicBlock],
    context_len: int,

    top_k: int,
) -> list[tuple[SymbolizedBasicBlock, float]]:
    # tokenize first context_len blocks
    sequence = (
        # add padding tokens if necessary
        [BasicBlockTokenizer.PADDING_TOKEN] * max(0, context_len - len(input))
        + [tokenizer.get_token(block) for block in input[-context_len:]]
    )

    input_tensor = torch.tensor(sequence, dtype=torch.long).to(model.device)

    predictions = model.predict_next_block(input_tensor, top_k=top_k)

    return [(tokenizer.get_block(tk), prob) for tk, prob in predictions]