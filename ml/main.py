import os
import random
from pathlib import Path
from typing import Literal, Optional

import torch

from bb_toolkit import TraceData, BasicBlockTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.dataset import analyze_sequence_stats, create_training_data, BasicBlockDataset
from ml.inference import make_inference
from ml.model import create_model, BasicBlockPredictor
from ml.train import train_epoch


def load_all_traces(traces_dir: os.PathLike) -> list[TraceData]:
    files = sorted(Path(traces_dir).glob("*.trace"), key=lambda f: f.stat().st_ctime)

    return [TraceData.from_binary_file(file) for file in files]

def load_model_for_inference(model_path: str, tokenizer_path: str, context_length: int, device: str) -> tuple[BasicBlockPredictor, BasicBlockTokenizer]:
    tokenizer = BasicBlockTokenizer.load_from_mapping(tokenizer_path)

    vocab_size = len(tokenizer)
    model = create_model(vocab_size, context_length=context_length).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model, tokenizer

def create_load_tokenizer(tokenizer_path: os.PathLike, traces_path: os.PathLike) -> tuple[BasicBlockTokenizer, list[list[int]]]:
    if os.path.exists(tokenizer_path):
        tokenizer = BasicBlockTokenizer.load_from_mapping(tokenizer_path)
    else:
        tokenizer = BasicBlockTokenizer()

    all_traces = load_all_traces(traces_path)

    prev_len = len(tokenizer)
    sequences: list[list[int]] = [tokenizer.process_trace(trace) for trace in all_traces]

    if len(tokenizer) > prev_len:
        tokenizer.save_mapping_to_file(tokenizer_path)

    return tokenizer, sequences

def setup_model(model_save_path: Optional[os.PathLike], context_len: int, vocab_size: int) -> BasicBlockPredictor:
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(vocab_size, context_length=context_len).to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=model.device))
        print(f"loaded model from {model_save_path}")

    return model


def train_model(
    model: BasicBlockPredictor,
    model_save_path: Optional[os.PathLike],
    epochs: int,
    train_data: DataLoader[BasicBlockDataset],
    val_data: DataLoader[BasicBlockDataset]
) -> None:
    optimizer = AdamW(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(model, optimizer, train_data, val_data)

        print(f"epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss and model_save_path is not None:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("saved new best model")

def main():
    model_save_path = "../run/best_model.pt"
    tokenizer_save_path = "../run/tokenizer.bin"
    trace_path = "../bulk_collect/traces"
    sequence_length = 16

    # 1. tokens
    tokenizer, sequences = create_load_tokenizer(tokenizer_save_path, trace_path)

    # 2. stats
    print(analyze_sequence_stats(sequences))

    # 3. partition data
    train_data, val_data, test_data = create_training_data(
        tokenized_sequences=sequences,
        sequence_length=sequence_length,
        validation_size=0.15,
        test_size=0.15,
        batch_size=8,
        seed=318,
    )

    model = setup_model(model_save_path, sequence_length, len(tokenizer))

    # 4. training
    # train_model(model, model_save_path, 3, train_data, val_data)

    # 5. inference
    infer_seq = random.choice(sequences)
    infer_loc = random.randint(sequence_length, len(infer_seq) - 1)

    context = infer_seq[:infer_loc]
    actual_next = infer_seq[infer_loc]

    prediction = make_inference(model, tokenizer, context, sequence_length, 5)

    print(f"prediction:\n{prediction[0][0]}\ncorrect ?: {actual_next in [pred for pred, _ in prediction]}")

if __name__ == "__main__":
    main()
