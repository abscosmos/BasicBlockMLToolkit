import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional, Any
import numpy as np
from collections import deque
import time

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from ml.model import OnlineBasicBlockPredictor
from ml.dataset import BasicBlockDataset, create_training_data
from ml.model import ModelConfig
from bb_toolkit import BasicBlockTokenizer


class OnlineLearner:
    """
    Manages incremental learning for the basic block predictor.
    Handles both initial training and lightweight online updates.
    """
    
    def __init__(
        self,
        model: OnlineBasicBlockPredictor,
        tokenizer: BasicBlockTokenizer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        initial_lr: float = 1e-4,
        incremental_lr: float = 1e-5,
        confidence_threshold: float = 0.1,
        buffer_size: int = 1000,
        update_frequency: int = 10
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        self.initial_optimizer = optimizer or AdamW(model.parameters(), lr=initial_lr)
        self.incremental_optimizer = AdamW(model.parameters(), lr=incremental_lr)
        
        # learning parameters
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency
        
        # performance tracking
        self.baseline_confidence: Optional[float] = None
        self.recent_confidences = deque(maxlen=50)
        self.trace_count: int = 0
        self.update_count: int = 0
        
        # experience replay buffer for incremental updates
        self.experience_buffer = deque(maxlen=buffer_size)

        self.training_history = {
            'initial_losses': [],
            'incremental_losses': [],
            'confidence_history': [],
            'update_timestamps': []
        }
    
    def initial_training(
        self,
        trace_sequences: list[list[int]],
        epochs: int = 10,
        batch_size: int = 8,
        validation_split: float = 0.15,
        test_split: float = 0.0,
        save_path: Optional[os.PathLike] = None,
        resume_from_epoch: int = 0,
    ) -> dict[str, Any]:
        """
        Perform initial training on historical trace data to establish base patterns.
        
        Args:
            trace_sequences: List of tokenized sequences from historical traces
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing (0.0 = no test split)
            save_path: Optional path to save best model
            resume_from_epoch: Epoch to resume training from (if loading from checkpoint)
            
        Returns:
            Training results and metrics
        """
        print(f"Starting initial training on {len(trace_sequences)} sequences...")
        
        train_loader, val_loader, test_loader = create_training_data(
            tokenized_sequences=trace_sequences,
            sequence_length=self.model.config.context_length,
            validation_size=validation_split,
            test_size=test_split,
            batch_size=batch_size,
            seed=42
        )
        
        self.model.train()
        best_val_loss = float('inf')

        epoch_pbar = tqdm(range(resume_from_epoch, epochs), desc="Initial Training", mininterval=0.5)

        for epoch in epoch_pbar:
            train_loss = self._train_epoch(train_loader, self.initial_optimizer)

            val_loss = self._validate_epoch(val_loader)
            
            self.training_history['initial_losses'].append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': self.model.get_vocab_size(),
                'timestamp': time.time()
            })

            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'vocab': self.model.get_vocab_size()
            })

            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                self._save_checkpoint(save_path, epoch, val_loss)
                epoch_pbar.write(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        self._establish_baseline_confidence(val_loader)
        
        # evaluate on test set if provided
        test_loss = None
        test_metrics = None
        if test_split > 0 and test_loader:
            test_loss = self._validate_epoch(test_loader)
            test_metrics = self._evaluate_test_performance(test_loader)
            print(f"Test evaluation: test_loss={test_loss:.4f}, test_accuracy={test_metrics['accuracy']:.4f}")
        
        print("Initial training completed!")

        result = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'final_vocab_size': self.model.get_vocab_size(),
            'training_history': self.training_history['initial_losses']
        }

        if test_loss is not None:
            result['test_loss'] = test_loss
            result['test_metrics'] = test_metrics
        
        return result
    
    def process_new_traces(
        self,
        new_sequences: list[list[int]],
        force_update: bool = False
    ) -> dict[str, Any]:
        """
        Process new trace sequences and decide whether to perform incremental updates.
        
        Args:
            new_sequences: List of new tokenized sequences
            force_update: Force an incremental update regardless of confidence
            
        Returns:
            Processing results and update decision
        """
        self.trace_count += len(new_sequences)
        
        # Add to experience buffer for potential future updates
        self.experience_buffer.extend(new_sequences)

        confidence = self._evaluate_confidence(new_sequences[:10])

        self.recent_confidences.append(confidence)
        self.training_history['confidence_history'].append({
            'trace_count': self.trace_count,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        should_update = (
            force_update or
            self._should_update_model() or
            (self.trace_count % self.update_frequency == 0)
        )
        
        result = {
            'processed_sequences': len(new_sequences),
            'current_confidence': confidence,
            'should_update': should_update,
            'total_traces_processed': self.trace_count,
            'vocab_size': self.model.get_vocab_size()
        }
        
        if should_update:
            update_result = self.incremental_update(new_sequences)
            result['update_result'] = update_result
            print(f"Performed incremental update #{self.update_count}")
        
        return result
    
    def incremental_update(
        self,
        new_sequences: list[list[int]],
        steps: int = 3,
        mix_ratio: float = 0.3
    ) -> dict[str, Any]:
        """
        Perform lightweight incremental updates on new data.
        
        Args:
            new_sequences: New sequences to learn from
            steps: Number of gradient steps to perform
            mix_ratio: Ratio of old data to mix with new data
            
        Returns:
            Update results and metrics
        """
        self.update_count += 1

        update_sequences = list(new_sequences)
        
        # mix in some representative samples from buffer to prevent forgetting
        if self.experience_buffer and mix_ratio > 0:
            buffer_sample_size = int(len(new_sequences) * mix_ratio)
            buffer_samples = list(self.experience_buffer)[-buffer_sample_size:]
            update_sequences.extend(buffer_samples)
        
        if len(update_sequences) == 0:
            return {'error': 'No sequences to update with'}
        
        try:
            train_loader, _, _ = create_training_data(
                tokenized_sequences=update_sequences,
                sequence_length=self.model.config.context_length,
                validation_size=0.0,
                test_size=0.0,
                batch_size=min(8, len(update_sequences)),
                seed=None
            )
        except ValueError as e:
            return {'error': f'Failed to create update dataset: {e}'}
        
        self.model.train()
        total_loss = 0.0
        vocab_before = self.model.get_vocab_size()
        
        for step in range(steps):
            step_loss = self._train_epoch(train_loader, self.incremental_optimizer, desc=f"Incremental Step {step+1}")
            total_loss += step_loss
        
        avg_loss = total_loss / steps
        vocab_after = self.model.get_vocab_size()
        
        self.training_history['incremental_losses'].append({
            'update_count': self.update_count,
            'trace_count': self.trace_count,
            'steps': steps,
            'avg_loss': avg_loss,
            'vocab_before': vocab_before,
            'vocab_after': vocab_after,
            'new_tokens': vocab_after - vocab_before,
            'timestamp': time.time()
        })
        self.training_history['update_timestamps'].append(time.time())
        
        return {
            'update_id': self.update_count,
            'steps_performed': steps,
            'avg_loss': avg_loss,
            'vocab_growth': vocab_after - vocab_before,
            'sequences_used': len(update_sequences),
            'new_sequences': len(new_sequences),
            'buffer_sequences': len(update_sequences) - len(new_sequences)
        }
    
    def _should_update_model(self) -> bool:
        """Determine if model needs incremental update based on confidence degradation."""
        if self.baseline_confidence is None or len(self.recent_confidences) < 5:
            return False
        
        recent_avg = np.mean(list(self.recent_confidences)[-10:])
        
        confidence_drop: float = self.baseline_confidence - recent_avg
        return confidence_drop > self.confidence_threshold
    
    def _evaluate_confidence(self, sequences: list[list[int]]) -> float:
        """Evaluate model confidence on a set of sequences."""
        if not sequences:
            return 0.0
        
        self.model.eval()
        total_confidence = 0.0
        total_predictions = 0
        
        with torch.no_grad():
            for seq in sequences[:10]:
                if len(seq) <= self.model.context_length:
                    continue
                
                # random slice for prediction
                start_idx = np.random.randint(0, len(seq) - self.model.context_length)
                context = seq[start_idx:start_idx + self.model.context_length - 1]
                target = seq[start_idx + self.model.context_length - 1]
                
                if len(context) == 0:
                    continue
                
                # Get predictions
                predictions = self.model.predict_next_block(
                    torch.tensor(context, device=self.model.device), 
                    top_k=5
                )
                
                # calc confidence based on top prediction probability
                if predictions:
                    confidence = predictions[0][1] # probability
                    total_confidence += confidence
                    total_predictions += 1
        
        return total_confidence / max(total_predictions, 1)
    
    def _evaluate_test_performance(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate comprehensive performance metrics on test set."""
        self.model.eval()
        
        total_predictions = 0
        correct_predictions = 0
        top5_correct = 0
        total_confidence = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(self.model.device)
                batch_targets = batch_targets.to(self.model.device)
                
                batch_size, seq_len = batch_inputs.shape
                
                for i in range(min(batch_size, 5)):
                    context = batch_inputs[i, :-1]
                    target = batch_targets[i, -1].item()
                    
                    predictions = self.model.predict_next_block(context, top_k=5)
                    if predictions:
                        # top-1
                        if predictions[0][0] == target:
                            correct_predictions += 1
                        
                        # top-5
                        predicted_tokens = [pred[0] for pred in predictions]
                        if target in predicted_tokens:
                            top5_correct += 1
                        
                        total_confidence += predictions[0][1]
                        total_predictions += 1
        
        if total_predictions == 0:
            return {'accuracy': 0.0, 'top5_accuracy': 0.0, 'avg_confidence': 0.0}
        
        return {
            'accuracy': correct_predictions / total_predictions,
            'top5_accuracy': top5_correct / total_predictions,
            'avg_confidence': total_confidence / total_predictions,
            'total_predictions': total_predictions
        }
    
    def _establish_baseline_confidence(self, val_loader: DataLoader):
        """Establish baseline confidence after initial training."""
        print("Establishing baseline confidence...")
        
        confidences = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_inputs, batch_targets in tqdm(val_loader, desc="Computing baseline confidence", leave=False):
                batch_size, seq_len = batch_inputs.shape
                
                for i in range(min(batch_size, 10)):
                    context = batch_inputs[i, :-1]
                    
                    predictions = self.model.predict_next_block(context, top_k=1)
                    if predictions:
                        confidences.append(predictions[0][1])
        
        if confidences:
            self.baseline_confidence = np.mean(confidences)
            print(f"Baseline confidence established: {self.baseline_confidence:.4f}")
        else:
            self.baseline_confidence = 0.5  # Default fallback
            print("Warning: Could not establish baseline confidence, using default")
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        desc: str = "Training"
    ) -> float:
        """Train for one epoch and return average loss."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=desc, leave=False)
        
        for batch_inputs, batch_targets in progress_bar:
            batch_inputs = batch_inputs.to(self.model.device)
            batch_targets = batch_targets.to(self.model.device)
            
            optimizer.zero_grad()

            # forward
            outputs = self.model(batch_inputs, labels=batch_targets)
            loss = outputs['loss']
            
            # backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.model.device)
                batch_targets = batch_targets.to(self.model.device)
                
                outputs = self.model(batch_inputs, labels=batch_targets)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab_size': len(self.tokenizer),
            'model_config': self.model.config,
            'epoch': epoch,
            'val_loss': val_loss,
            'training_history': self.training_history,
            'baseline_confidence': self.baseline_confidence,
            'update_count': self.update_count,
            'trace_count': self.trace_count,
            'initial_optimizer_state_dict': self.initial_optimizer.state_dict(),
            'incremental_optimizer_state_dict': self.incremental_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load model checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location=self.model.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.baseline_confidence = checkpoint.get('baseline_confidence', None)
        self.update_count = checkpoint.get('update_count', 0)
        self.trace_count = checkpoint.get('trace_count', 0)
        
        # restore optimizer states if available
        if 'initial_optimizer_state_dict' in checkpoint:
            self.initial_optimizer.load_state_dict(checkpoint['initial_optimizer_state_dict'])
        if 'incremental_optimizer_state_dict' in checkpoint:
            self.incremental_optimizer.load_state_dict(checkpoint['incremental_optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'tokenizer_vocab_size': checkpoint.get('tokenizer_vocab_size', 0),
            'model_vocab_size': self.model.get_vocab_size()
        }
    
    def get_training_stats(self) -> dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'trace_count': self.trace_count,
            'update_count': self.update_count,
            'current_vocab_size': self.model.get_vocab_size(),
            'baseline_confidence': self.baseline_confidence,
            'recent_confidence': np.mean(list(self.recent_confidences)) if self.recent_confidences else None,
            'buffer_size': len(self.experience_buffer),
            'training_history': self.training_history
        }