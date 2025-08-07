import torch
from bb_toolkit import BasicBlockTokenizer, SymbolizedBasicBlock
from ml.model import BasicBlockPredictor

def make_inference(
    model: BasicBlockPredictor,
    tokenizer: BasicBlockTokenizer,

    input_seq: list[SymbolizedBasicBlock] | list[int],
    context_len: int,

    top_k: int,
) -> list[tuple[SymbolizedBasicBlock, float]]:
    if len(input_seq) == 0 or isinstance(input_seq[0], int):
        input_tokens: list[int] = input_seq
    else:
        input_tokens: list[int] = [tokenizer.get_token(block) for block in input_seq]

    # tokenize first context_len blocks
    sequence: list[int] = (
        # add padding tokens if necessary
        [BasicBlockTokenizer.PADDING_TOKEN] * max(0, context_len - len(input_seq))
        + input_tokens[-context_len:]
    )

    input_tensor = torch.tensor(sequence, dtype=torch.long).to(model.device)

    predictions = model.predict_next_block(input_tensor, top_k=top_k)

    return [(tokenizer.get_block(tk), prob) for tk, prob in predictions]