import torch
from typing import Optional, Any
import os
from pathlib import Path

from bb_toolkit import TraceData, BasicBlockTokenizer, BasicBlock, SymbolizedBasicBlock
from ml.model import OnlineBasicBlockPredictor, ModelConfig
from ml.online_trainer import OnlineLearner

class BasicBlockPredictor:
    """
    High-level interface for online basic block prediction.
    Handles model loading, trace processing, and incremental learning.
    """
    
    def __init__(
        self,
        model_path: Optional[os.PathLike] = None,
        tokenizer_path: Optional[os.PathLike] = None,
        auto_update: bool = True,
        update_threshold: float = 0.1,
        update_frequency: int = 10,
        device: Optional[torch.device] = None,
        config: Optional[ModelConfig] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.auto_update = auto_update

        # cache for repeated contexts
        self._prediction_cache = {}

        # load or create tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = BasicBlockTokenizer.load_from_mapping(tokenizer_path)
            print(f"Loaded tokenizer with vocab size {len(self.tokenizer)}")
        else:
            self.tokenizer = BasicBlockTokenizer()
        
        # load or create model
        if model_path and os.path.exists(model_path):
            self.model, self.learner = self._load_model_and_learner(
                model_path, update_threshold, update_frequency
            )
            print(f"Loaded model: {self.get_model_stats()}")
        else:
            # create fresh model if no saved model exists
            initial_vocab_size = max(1000, len(self.tokenizer))

            self.model = OnlineBasicBlockPredictor(
                initial_vocab_size=initial_vocab_size,
                config=config,
                device=self.device
            ).to(self.device)

            print(f"Created OL model with {sum(p.numel() for p in self.model.parameters()):,} params")
            print(f"Initial vocabulary size: {initial_vocab_size}")
            print(f"Config: {self.model.config}")
            
            self.learner = OnlineLearner(
                self.model,
                self.tokenizer,
                confidence_threshold=update_threshold,
                update_frequency=update_frequency
            )
        
        self.model.eval()
    
    def predict_next_block_from_sequence(
        self,
        input_sequence: list[int] | list[SymbolizedBasicBlock] | list[BasicBlock],
        top_k: int = 5,
        use_cache: bool = True
    ) -> list[tuple[int, float]]:
        """
        Predict next blocks from a sequence of tokens or symbolized basic blocks.
        
        Args:
            input_sequence: Either list of token IDs or list of SymbolizedBasicBlock objects
            top_k: Number of top predictions to return
            use_cache: Whether to use prediction cache
            
        Returns:
            List of prediction dictionaries with token_id, probability, and block_signature
        """
        # Handle different input types
        if not input_sequence:
            return []
            
        if isinstance(input_sequence[0], int):
            token_sequence = input_sequence
        elif isinstance(input_sequence[0], (BasicBlock, SymbolizedBasicBlock)):
            # Handle both BasicBlock and SymbolizedBasicBlock objects
            token_sequence = []
            for block in input_sequence:
                symbolized_block = block.symbolize() if isinstance(block, BasicBlock) else block

                token_sequence.append(self.tokenizer.get_token(symbolized_block))
            
            if not token_sequence:
                return []
        else:
            raise TypeError(f"Unsupported input type: {type(input_sequence[0])}. Expected int, BasicBlock, or SymbolizedBasicBlock.")
        
        # Take the most recent context_length tokens
        context_length = self.model.config.context_length
        if len(token_sequence) > context_length:
            context = token_sequence[-context_length:]
        else:
            # add padding
            padding_needed = context_length - len(token_sequence)
            context = [BasicBlockTokenizer.PADDING_TOKEN] * padding_needed + token_sequence
        
        # Check cache first
        if use_cache:
            cache_key = hash(tuple(context))
            if cache_key in self._prediction_cache:
                return self._prediction_cache[cache_key]
        
        # make prediction
        context_tensor = torch.tensor(context, dtype=torch.long, device=self.device)
        predictions = self.model.predict_next_block(context_tensor, top_k=top_k)
        
        # cache result
        if use_cache and predictions:
            self._prediction_cache[cache_key] = predictions

        return predictions

    def process_new_traces(
        self,
        new_traces: list[TraceData],
        force_update: bool = False
    ) -> dict[str, Any]:
        """
        Process new trace data and potentially trigger model updates.
        
        Args:
            new_traces: List of new trace data
            force_update: Force incremental update regardless of confidence
            
        Returns:
            Processing results including update decisions
        """
        if not new_traces:
            return {'error': 'No traces provided'}
        
        # tokenize all new traces
        new_sequences = []
        for trace in new_traces:
            sequence = self.tokenizer.process_trace(trace)
            if sequence:
                new_sequences.append(sequence)
        
        if not new_sequences:
            return {'error': 'No valid sequences from traces'}
        
        # clear cache if vocabulary expanded
        vocab_before = len(self.tokenizer)
        
        # process through learner if auto_update enabled
        if self.auto_update:
            result = self.learner.process_new_traces(new_sequences, force_update)
            
            # clear cache if vocabulary grew
            vocab_after = len(self.tokenizer)
            if vocab_after > vocab_before:
                self._prediction_cache.clear()
            
            return result
        else:
            # just add to buffer without triggering updates
            self.learner.experience_buffer.extend(new_sequences)
            return {
                'processed_sequences': len(new_sequences),
                'auto_update_disabled': True,
                'vocab_size': len(self.tokenizer)
            }
    
    def update_model(self, new_traces: list[TraceData], steps: int = 3) -> dict[str, Any]:
        """
        Manually trigger incremental model updates.
        
        Args:
            new_traces: New trace data to learn from  
            steps: Number of gradient steps for update
            
        Returns:
            Update results and metrics
        """
        new_sequences = []
        for trace in new_traces:
            sequence = self.tokenizer.process_trace(trace)
            if sequence:
                new_sequences.append(sequence)
        
        if not new_sequences:
            return {'error': 'No valid sequences to update with'}
        
        self._prediction_cache.clear()
        
        result = self.learner.incremental_update(new_sequences, steps=steps)
        
        return result
    
    def save_model(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Save model and tokenizer to disk.
        
        Args:
            model_path: Path to save model checkpoint
            tokenizer_path: Path to save tokenizer (optional)
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.learner._save_checkpoint(model_path, epoch=0, val_loss=0.0)
        
        if tokenizer_path:
            tokenizer_dir = Path(tokenizer_path).parent
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_mapping_to_file(tokenizer_path)

    def get_model_stats(self) -> dict[str, Any]:
        """Get comprehensive model and training statistics."""
        return {
            'model_vocab_size': self.model.get_vocab_size(),
            'tokenizer_vocab_size': len(self.tokenizer),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'cache_size': len(self._prediction_cache),
            'training_stats': self.learner.get_training_stats() if self.learner else None
        }
    
    def clear_cache(self):
        """Clear prediction cache to free memory."""
        self._prediction_cache.clear()
    
    def _load_model_and_learner(
        self, 
        model_path: os.PathLike,
        confidence_threshold: float,
        update_frequency: int
    ) -> tuple[OnlineBasicBlockPredictor, OnlineLearner]:
        """Load model and create learner from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # get vocab size from checkpoint or tokenizer
        saved_vocab_size = checkpoint.get('tokenizer_vocab_size', len(self.tokenizer))

        if 'model_config' not in checkpoint:
            raise ValueError("Checkpoint missing required 'model_config'. Cannot load model architecture.")
        
        config: ModelConfig = checkpoint['model_config']

        # create model with appropriate vocab size
        model = OnlineBasicBlockPredictor(
            device=self.device,
            initial_vocab_size=max(1000, saved_vocab_size, len(self.tokenizer)),
            config=config,
        ).to(self.device)
        
        # ensure model components match saved state
        state_dict = checkpoint['model_state_dict']
        
        # check if we need to expand embedding for saved state
        if 'embedding.current_vocab_size' in state_dict:
            target_vocab_size = state_dict['embedding.current_vocab_size'].item()
            model.embedding.expand_vocabulary(target_vocab_size)
        
        # ensure output projection exists with correct size
        if 'output_projection.weight' in state_dict:
            output_size = state_dict['output_projection.weight'].shape[0]
            model._ensure_output_projection(output_size)
        
        # load model state with strict=False to handle missing/extra keys
        model.load_state_dict(state_dict, strict=False)
        
        # create learner and restore training state
        learner = OnlineLearner(
            model,
            self.tokenizer,
            confidence_threshold=confidence_threshold,
            update_frequency=update_frequency
        )
        
        # restore learner state from checkpoint
        learner.training_history = checkpoint.get('training_history', learner.training_history)
        learner.baseline_confidence = checkpoint.get('baseline_confidence', None)
        learner.update_count = checkpoint.get('update_count', 0)
        learner.trace_count = checkpoint.get('trace_count', 0)
        
        return model, learner