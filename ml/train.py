import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml.model import BasicBlockPredictor
from ml.dataset import BasicBlockDataset

from typing import Literal

def train_epoch(
    model: BasicBlockPredictor,

    optimizer: Optimizer,
    device: Literal["cpu", "cuda"],

    training_data: DataLoader[BasicBlockDataset],
    validation_data: DataLoader[BasicBlockDataset],
):
    model.train()
    train_loss = 0.0

    total_train_batches = len(training_data)
    progress_interval = max(1, total_train_batches // 100)

    for batch_idx, (inputs, targets) in enumerate(training_data):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

        # print progress every 1%
        if (batch_idx + 1) % progress_interval == 0:
            progress = (batch_idx + 1) / total_train_batches * 100
            current_loss = loss.item()
            print(f"  progress: {progress:.0f}% - batch loss: {current_loss:.4f}")

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