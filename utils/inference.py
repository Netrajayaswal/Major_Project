#!/usr/bin/env python3
"""
Inference Pipeline for Text-to-Sign Translation
Converts text input to skeleton-based sign language animation videos

Features:
- Real-time text-to-sign translation
- Skeleton video generation
- Multiple output formats (NPY, MP4, GIF)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import MODEL_CONFIG, OUTPUT_CONFIG, SAMPLE_GLOSS_VOCABULARY
from models.text_to_sign_model import TextToSignModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE TOKENIZER
# =============================================================================

class SimpleTokenizer:
    """Simple tokenizer for text processing."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize with special tokens
        self._add_token(self.pad_token, self.pad_token_id)
        self._add_token(self.unk_token, self.unk_token_id)
        self._add_token(self.sos_token, self.sos_token_id)
        self._add_token(self.eos_token, self.eos_token_id)
    
    def _add_token(self, token: str, idx: int):
        """Add token to vocabulary."""
        self.word_to_idx[token] = idx
        self.idx_to_word[idx] = token
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        text = text.lower().strip()
        tokens = text.split()
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text string
            add_special_tokens: Add SOS and EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate to max_length
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        max_length = max_length or self.max_length
        
        tokens = self._tokenize(text)
        
        # Convert to indices
        indices = []
        if add_special_tokens:
            indices.append(self.sos_token_id)
        
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                # Hash unknown tokens to vocab range
                idx = hash(token) % (self.vocab_size - 4) + 4
                indices.append(idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
        
        if add_special_tokens:
            indices.append(self.eos_token_id)
        
        # Truncate
        if truncation and len(indices) > max_length:
            indices = indices[:max_length-1] + [self.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(indices)
        
        # Pad
        if padding and len(indices) < max_length:
            padding_length = max_length - len(indices)
            indices = indices + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor([indices], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
        }
    
    def decode(self, ids: torch.Tensor) -> str:
        """Decode token indices to text."""
        tokens = []
        for idx in ids:
            idx = idx.item()
            if idx in self.idx_to_word:
                if idx > 3:  # Skip special tokens
                    tokens.append(self.idx_to_word[idx])
        return ' '.join(tokens)
    
    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Call encode method."""
        return self.encode(text, **kwargs)


# =============================================================================
# GLOSS VOCABULARY
# =============================================================================

class GlossVocabulary:
    """Gloss vocabulary for sign language."""
    
    def __init__(self):
        self.gloss_to_idx = SAMPLE_GLOSS_VOCABULARY.copy()
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
    
    def encode(self, gloss: str) -> int:
        """Convert gloss to index."""
        return self.gloss_to_idx.get(gloss.upper(), self.gloss_to_idx.get("<UNK>", 3))
    
    def decode(self, idx: int) -> str:
        """Convert index to gloss."""
        return self.idx_to_gloss.get(idx, "<UNK>")
    
    def __len__(self) -> int:
        return len(self.gloss_to_idx)


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

class TextToSignInference:
    """
    Complete inference pipeline for text-to-sign translation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            config: Model configuration
        """
        # Device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Config
        self.config = config or self._default_config()
        
        # Tokenizer and vocabulary
        self.tokenizer = SimpleTokenizer(
            vocab_size=self.config.get('vocab_size', 10000),
            max_length=self.config.get('max_text_length', 128),
        )
        self.gloss_vocab = GlossVocabulary()
        
        # Model
        self.model = TextToSignModel(self.config)
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Inference pipeline initialized on {self.device}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            'vocab_size': 10000,
            'text_embedding_dim': 768,
            'text_hidden_dim': 768,
            'text_encoder_layers': 4,
            'gloss_vocab_size': 100,
            'gloss_hidden_dim': 512,
            'gloss_encoder_layers': 4,
            'gloss_decoder_layers': 4,
            'gloss_embed_dim': 256,
            'pose_hidden_dim': 512,
            'pose_layers': 6,
            'refine_hidden_dim': 256,
            'refine_layers': 9,
            'pose_dim': 543,
            'pose_coords': 3,
            'num_heads': 8,
            'max_frames': 60,
            'max_text_length': 128,
            'temporal_kernel': 9,
            'dropout': 0.1,
        }
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        logger.info("Model loaded successfully")
    
    def translate(
        self,
        text: str,
        max_frames: int = 60,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Translate text to sign language skeleton animation.
        
        Args:
            text: Input text string
            max_frames: Maximum number of frames to generate
            return_intermediate: Return intermediate results (gloss, unrefined poses)
        
        Returns:
            Dictionary containing:
                - skeleton: numpy array of skeleton poses [frames, keypoints, coords]
                - gloss_text: predicted gloss sequence
                - gloss_ids: gloss token indices
        """
        logger.info(f"Translating: '{text}'")
        
        with torch.no_grad():
            # Tokenize text
            inputs = self.tokenizer(text)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate gloss sequence
            gloss_outputs = self.model.text_to_gloss(
                input_ids, attention_mask, max_length=20
            )
            
            # Get generated gloss IDs
            if 'generated_ids' in gloss_outputs:
                gloss_ids = gloss_outputs['generated_ids']
            else:
                # Fallback: create simple gloss from input
                gloss_ids = self._create_simple_gloss(text)
            
            # Decode gloss
            gloss_text = self.model.text_to_gloss.decode_gloss(
                gloss_ids, self.gloss_vocab.idx_to_gloss
            )
            
            logger.info(f"Generated gloss: {gloss_text}")
            
            # Generate poses
            pose_outputs = self.model.gloss_to_pose(
                gloss_ids, target_frames=max_frames
            )
            
            # Refine poses
            refined_outputs = self.model.pose_refiner(pose_outputs['poses'])
            final_poses = refined_outputs['refined_poses']
            
            # Convert to numpy
            skeleton = final_poses[0].cpu().numpy()  # [frames, keypoints, coords]
        
        result = {
            'skeleton': skeleton,
            'gloss_text': gloss_text[0] if gloss_text else "",
            'gloss_ids': gloss_ids[0].cpu().numpy() if gloss_ids is not None else None,
        }
        
        if return_intermediate:
            result['intermediate'] = {
                'unrefined_poses': pose_outputs['poses'][0].cpu().numpy(),
                'durations': pose_outputs.get('durations', None),
            }
        
        logger.info(f"Generated {skeleton.shape[0]} frames")
        
        return result
    
    def _create_simple_gloss(self, text: str) -> torch.Tensor:
        """Create simple gloss sequence from text."""
        words = text.lower().split()
        gloss_ids = []
        
        for word in words:
            if word.upper() in self.gloss_vocab.gloss_to_idx:
                gloss_ids.append(self.gloss_vocab.encode(word))
            else:
                # Use word hash as fallback
                gloss_ids.append(hash(word) % 100 + 10)
        
        # Pad/truncate to fixed length
        max_len = 20
        if len(gloss_ids) < max_len:
            gloss_ids = gloss_ids + [0] * (max_len - len(gloss_ids))
        else:
            gloss_ids = gloss_ids[:max_len]
        
        return torch.tensor([gloss_ids], dtype=torch.long, device=self.device)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_sign_language_video(
    input_text: str,
    model: Optional[TextToSignModel] = None,
    model_path: Optional[str] = None,
    output_path: str = "output_skeleton.mp4",
    max_frames: int = 60,
    fps: int = 25,
) -> Tuple[np.ndarray, str]:
    """
    Generate sign language video from text.
    
    Args:
        input_text: Input text string
        model: Pre-loaded model (optional)
        model_path: Path to model checkpoint (optional)
        output_path: Path to save output video
        max_frames: Maximum number of frames
        fps: Frames per second
    
    Returns:
        Tuple of (skeleton_data, video_path)
    """
    # Initialize inference
    inference = TextToSignInference(model_path=model_path) if model is None else None
    
    if inference is not None:
        result = inference.translate(input_text, max_frames=max_frames)
        skeleton = result['skeleton']
    else:
        # Use provided model
        tokenizer = SimpleTokenizer()
        gloss_vocab = GlossVocabulary()
        
        with torch.no_grad():
            inputs = tokenizer(input_text)
            generated = model.generate(input_text, tokenizer, gloss_vocab.idx_to_gloss, max_frames)
            skeleton = generated['poses'][0].cpu().numpy()
    
    # Save skeleton data
    skeleton_path = output_path.replace('.mp4', '.npy')
    np.save(skeleton_path, skeleton)
    logger.info(f"Skeleton data saved to: {skeleton_path}")
    
    # Create video
    from utils.visualization import create_skeleton_video
    create_skeleton_video(skeleton, output_path, fps=fps)
    
    return skeleton, output_path


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-Sign Inference")
    parser.add_argument('--text', type=str, default="Hello, how are you?",
                       help='Input text to translate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./outputs/output_skeleton.mp4',
                       help='Output video path')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum number of frames')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inference = TextToSignInference(model_path=args.model_path)
    
    # Translate
    result = inference.translate(args.text, max_frames=args.max_frames)
    
    print(f"\nInput: {args.text}")
    print(f"Generated gloss: {result['gloss_text']}")
    print(f"Skeleton shape: {result['skeleton'].shape}")
    
    # Save outputs
    np.save(args.output.replace('.mp4', '.npy'), result['skeleton'])
    print(f"\nSkeleton saved to: {args.output.replace('.mp4', '.npy')}")
    
    print("\n✅ Inference complete!")
