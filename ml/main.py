import os
from pathlib import Path
from typing import Literal

import torch

from bb_toolkit import TraceData, BasicBlockTokenizer
from torch.optim import AdamW

from ml.dataset import analyze_sequence_stats, create_training_data
from ml.model import create_model
from ml.train import train_epoch


def load_all_traces(traces_dir: os.PathLike) -> list[TraceData]:
    files = sorted(Path(traces_dir).glob("*.trace"), key=lambda f: f.stat().st_ctime)

    return [TraceData.from_binary_file(file) for file in files]

def main():
    model_save_path = "../run/best_model.pt"
    tokenizer_save_path = "../run/tokenizer.bin"

    # 1. load traces
    trace_path = "../bulk_collect/traces"
    all_traces = load_all_traces(trace_path)

    print(f"Loaded {len(all_traces)} from {trace_path}.")

    # 2. tokenize
    if os.path.exists(tokenizer_save_path):
        tokenizer = BasicBlockTokenizer.load_from_mapping(tokenizer_save_path)
    else:
        tokenizer = BasicBlockTokenizer()

    prev_len = len(tokenizer)
    sequences: list[list[int]] = [tokenizer.process_trace(trace) for trace in all_traces]

    if len(tokenizer) > prev_len:
        tokenizer.save_mapping_to_file(tokenizer_save_path)

    # 3. stats
    print(analyze_sequence_stats(sequences))

    # 4. partition data
    sequence_length = 16
    train_data, val_data, test_data = create_training_data(
        tokenized_sequences=sequences,
        sequence_length=sequence_length,
        validation_size=0.15,
        test_size=0.15,
        batch_size=8,
    )

    # 5. training
    num_epochs = 2

    vocab_size = len(tokenizer)
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(vocab_size, context_length=sequence_length).to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"loaded model from {model_save_path}")

    optimizer = AdamW(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, val_loss = train_epoch(model, optimizer, device, train_data, val_data)

        print(f"epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("saved new best model")


if __name__ == "__main__":
    main()
