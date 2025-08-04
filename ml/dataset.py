import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

class BasicBlockDataset(Dataset):
    def __init__(self, tokenized_sequences: List[List[int]], context_len: int):
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
    tokenized_sequences: List[List[int]],
    sequence_length: int = 64,
    test_size: float = 0.2,
    batch_size: int = 32,
    pad_token_id: int = 0
) -> Tuple[DataLoader, DataLoader]:
    
    # filter sequences to ensure they are long enough
    min_length = sequence_length + 1
    valid_sequences = [seq for seq in tokenized_sequences if len(seq) >= min_length]

    print(f"Found {len(valid_sequences)} valid sequences from {len(tokenized_sequences)} total")

    if not valid_sequences:
        raise ValueError(f"No sequences long enough (minimum {min_length} tokens)")

    # split sequences into train/validation
    train_sequences, val_sequences = train_test_split(
        valid_sequences,
        test_size=test_size,
        random_state=42
    )
    
    print(f"{len(train_sequences)} training, {len(val_sequences)} validation")

    # create datasets
    train_dataset = BasicBlockDataset(train_sequences, sequence_length)
    val_dataset = BasicBlockDataset(val_sequences, sequence_length)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    return train_loader, val_loader

def analyze_sequence_stats(tokenized_sequences: List[List[int]]) -> dict:
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

def prepare_data_loaders(
    trace_file_path: str,
    sequence_length: int = 64,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, 'BasicBlockTokenizer']:
    print("Loading trace data...")
    trace_data = TraceData.from_binary_file(trace_file_path)

    tokenizer = BasicBlockTokenizer()
    tokenized_sequence = tokenizer.process_trace(trace_data)

    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Sequence length: {len(tokenized_sequence):,} tokens")

    stats = analyze_sequence_stats([tokenized_sequence])
    print(stats)

    print("Creating DataLoaders...")
    train_loader, val_loader = create_training_data(
        [tokenized_sequence],
        sequence_length=sequence_length,
        batch_size=batch_size,
        pad_token_id=BasicBlockTokenizer.get_pad_token()
    )

    return train_loader, val_loader, tokenizer