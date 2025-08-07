from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class BasicBlockDataset(Dataset):
    def __init__(self, tokenized_sequences: list[list[int]], context_len: int):
        self.context_len = context_len
        self.samples = []

        # Create sliding window samples from all sequences
        for sequence in tokenized_sequences:
            for i in range(len(sequence) - context_len):
                input_seq = sequence[i:i + context_len]
                target_seq = sequence[i + 1:i + context_len + 1]
                self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def create_training_data(
    tokenized_sequences: list[list[int]],
    sequence_length: int,

    validation_size: float = 0.15,
    test_size: float = 0.15,

    batch_size: int = 32,

    seed: Optional[int] = None,
) -> tuple[
    DataLoader[BasicBlockDataset],
    DataLoader[BasicBlockDataset],
    DataLoader[BasicBlockDataset]
]:
    assert validation_size + test_size < 1, "no data to train with"
    
    # filter sequences to ensure they are long enough
    min_length = sequence_length + 1
    valid_sequences = [seq for seq in tokenized_sequences if len(seq) >= min_length]

    print(f"Found {len(valid_sequences)} valid sequences from {len(tokenized_sequences)} total")

    if not valid_sequences:
        raise ValueError(f"No sequences long enough (minimum {min_length} tokens)")

    # split sequences into train/validation
    train_sequences, rest_sequences = train_test_split(
        valid_sequences,
        test_size=validation_size + test_size,
        random_state=seed,
        shuffle=True
    )

    val_sequences, test_sequences = train_test_split(
        rest_sequences,
        test_size=test_size / (validation_size + test_size),
        random_state=seed,
        shuffle=True
    )
    
    print(f"{len(train_sequences)=}, {len(val_sequences)=}, {len(test_sequences)=}")

    # create datasets
    train_dataset = BasicBlockDataset(train_sequences, sequence_length)
    val_dataset = BasicBlockDataset(val_sequences, sequence_length)
    test_dataset = BasicBlockDataset(test_sequences, sequence_length)

    print(f"{len(train_dataset)=}, {len(val_dataset)=}, {len(test_dataset)=}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with FFI types
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

def analyze_sequence_stats(tokenized_sequences: list[list[int]]) -> dict:
    lengths = [len(seq) for seq in tokenized_sequences]

    stats = {
        'num_sequences': len(tokenized_sequences),
        'total_tokens': sum(lengths),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0,
        'length_25th': np.percentile(lengths, 25),
        'length_75th': np.percentile(lengths, 75),
        'length_95th': np.percentile(lengths, 95)
    }

    return stats