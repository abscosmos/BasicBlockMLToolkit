import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml.model import BasicBlockPredictor
from ml.dataset import BasicBlockDataset

from tqdm import tqdm

from typing import Literal
import sys

def train_epoch(
    model: BasicBlockPredictor,

    optimizer: Optimizer,
    device: Literal["cpu", "cuda"],

    training_data: DataLoader[BasicBlockDataset],
    validation_data: DataLoader[BasicBlockDataset],
):
    model.train()
    train_loss = 0.0

    sys.stdout.flush()
    progress_bar = tqdm(total=len(training_data), desc="Training", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(training_data):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        progress_bar.update(1)

    train_loss /= len(training_data.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in validation_data:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(validation_data.dataset)

    return train_loss, val_loss