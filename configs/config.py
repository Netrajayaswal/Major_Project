#!/usr/bin/env python3
"""
Configuration for Text-to-Sign Translation Model
Optimized for 4GB VRAM + 16GB RAM hardware constraints
"""

import torch
from datetime import datetime

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================

HARDWARE_CONFIG = {
    "vram": "4GB",
    "ram": "16GB",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # Hardware Constraints
    "vram": "4GB",
    "ram": "16GB",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Batch Processing (CRITICAL for 4GB VRAM)
    "batch_size": 2,                    # Very small batch due to VRAM limit
    "gradient_accumulation_steps": 8,   # Simulate batch_size=16 (2*8)
    "num_workers": 2,                   # Dataloader workers (limited by RAM)
    "pin_memory": False,                # Disable to save RAM
    
    # Videos Per Training Session
    "videos_per_class": 10,             # Use only 10 videos per class initially
    "total_classes": 20,                # Start with 20 sign classes
    "total_videos_loaded": 200,         # 20 classes × 10 videos = 200 videos
    
    # Checkpoint Configuration
    "checkpoint_frequency": 30,         # Save every 30 minutes
    "checkpoint_dir": "./checkpoints/",
    "keep_last_n_checkpoints": 5,       # Keep only last 5 to save disk space
    "auto_resume": True,                # Auto-resume from last checkpoint
    
    # Training Schedule
    "training_start_time": "22:00",     # 10:00 PM
    "training_end_time": "05:00",       # 5:00 AM
    "auto_shutdown_buffer": 5,          # Stop 5 min before end time
    "max_epochs": 100,
    "early_stopping_patience": 15,
    
    # Model Optimization
    "mixed_precision": True,            # FP16 training (saves VRAM)
    "gradient_checkpointing": True,     # Trade compute for memory
    "max_sequence_length": 60,          # Max frames per video (2.4 sec at 25fps)
    
    # Optimizer
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    
    # Learning Rate Scheduler
    "scheduler_type": "cosine_annealing_warm_restarts",
    "scheduler_T_0": 10,
    "scheduler_T_mult": 2,
    "warmup_steps": 500,
    
    # Loss Configuration
    "label_smoothing": 0.1,
    "reconstruction_loss_weight": 0.95,
    "smoothness_loss_weight": 0.05,
    "kinematic_loss_weight": 0.1,
}

# =============================================================================
# TEXT INPUT PROCESSING CONFIGURATION
# =============================================================================

TEXT_INPUT_CONFIG = {
    # Real-Time Text Input
    "input_type": "real_time_text",
    "supported_languages": ["English", "Hindi", "Spanish"],
    "max_input_length": 128,            # Max characters
    
    # Text Encoder Model
    "text_encoder": "distilbert-base-multilingual-cased",  # Lighter than mBERT
    "embedding_dim": 768,
    "freeze_encoder": False,            # Fine-tune on sign language data
    
    # Preprocessing
    "lowercase": True,
    "remove_punctuation": False,        # Keep for grammar understanding
    "tokenizer": "bert-base-multilingual-cased",
}

# =============================================================================
# TEXT-TO-GLOSS CONFIGURATION
# =============================================================================

TEXT_TO_GLOSS_CONFIG = {
    # Seq2Seq Transformer (Encoder-Decoder)
    "architecture": "transformer",
    "encoder_layers": 4,                # Reduced from 6 (memory optimization)
    "decoder_layers": 4,
    "attention_heads": 8,
    "hidden_dim": 512,                  # Reduced from 768
    "dropout": 0.1,
    "feed_forward_dim": 2048,
    
    # Training
    "loss_function": "cross_entropy_with_label_smoothing",
    "label_smoothing": 0.1,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    
    # Inference
    "beam_search_width": 5,
    "max_gloss_length": 20,
    
    # Vocabulary
    "vocab_size": 1000,                 # Approximate gloss vocabulary
    "pad_token_id": 0,
    "sos_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
}

# =============================================================================
# GLOSS-TO-SKELETON-POSE CONFIGURATION
# =============================================================================

GLOSS_TO_POSE_CONFIG = {
    # Motion Generation Model
    "architecture": "motion_transformer",
    "gloss_embedding_dim": 256,
    "pose_dim": 543,                    # 33 body + 21*2 hands + 468 face
    "temporal_layers": 6,
    "attention_heads": 8,
    "hidden_dim": 512,
    "dropout": 0.1,
    
    # Output Configuration
    "output_fps": 25,                   # Match dataset FPS
    "output_format": "mediapipe_holistic",
    "coordinate_system": "normalized",  # [0, 1] range
    
    # Skeleton Structure
    "keypoints": {
        "body_landmarks": 33,           # MediaPipe body pose
        "left_hand_landmarks": 21,
        "right_hand_landmarks": 21,
        "face_landmarks": 468,
        "total": 543
    },
    
    # Motion Constraints
    "max_motion_speed": 2.0,            # Limit unrealistic fast movements
    "smooth_window": 5,                 # Frames for smoothing
    "kinematic_loss_weight": 0.1,       # Penalize impossible poses
    
    # Temporal Modeling
    "max_sequence_length": 60,          # Max frames
    "min_sequence_length": 10,          # Min frames
}

# =============================================================================
# POSE REFINEMENT CONFIGURATION
# =============================================================================

REFINEMENT_CONFIG = {
    # Graph Convolutional Network (ST-GCN)
    "architecture": "st_gcn",
    "graph_structure": "skeleton_graph",
    "spatial_kernel_size": 1,
    "temporal_kernel_size": 9,
    "num_layers": 9,
    "hidden_dim": 256,
    "dropout": 0.1,
    
    # Smoothing Filters
    "temporal_filter": "savitzky_golay",
    "filter_window": 7,
    "filter_order": 3,
    
    # Quality Metrics
    "smoothness_loss_weight": 0.05,
    "reconstruction_loss_weight": 0.95,
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    # Video Generation
    "output_format": "skeleton_visualization",
    "render_type": "opencv_draw",       # or "matplotlib_animation"
    "video_codec": "mp4v",
    "output_resolution": [1920, 1080],
    "fps": 25,
    
    # Visualization Settings
    "skeleton_color": "green",
    "body_color": (0, 255, 0),          # BGR format
    "left_hand_color": (255, 0, 0),     # Blue
    "right_hand_color": (0, 0, 255),    # Red
    "face_color": (255, 255, 255),      # White
    "line_thickness": 2,
    "keypoint_radius": 5,
    "show_connections": True,
    "background_color": "black",
    
    # Save Format
    "save_skeleton_data": True,         # Save .npy file
    "save_video_file": True,            # Save .mp4 file
    "save_gif": False,                  # Optional GIF export
    
    # Output Paths
    "skeleton_output_dir": "./outputs/skeletons/",
    "video_output_dir": "./outputs/videos/",
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    # Paths
    "data_root": "./data/",
    "organized_classes_dir": "./data/organized_classes/",
    "train_subset_dir": "./data/train_subset/",
    "val_subset_dir": "./data/val_subset/",
    "test_subset_dir": "./data/test_subset/",
    
    # Data Loading
    "videos_per_class": 10,
    "max_frames": 60,
    "skeleton_format": "mediapipe_holistic",
    
    # Augmentation
    "use_augmentation": True,
    "augmentation_ratio": 0.5,          # Ratio of augmented to regular samples
    
    # Train/Val/Test Split
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Metadata
    "metadata_filename": "metadata.json",
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    # Logging
    "log_dir": "./logs/",
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_console": True,
    
    # Tensorboard
    "use_tensorboard": True,
    "tensorboard_dir": "./logs/tensorboard/",
    
    # Metrics
    "log_every_n_steps": 10,
    "validate_every_n_epochs": 1,
    "save_every_n_epochs": 5,
    
    # Progress
    "show_progress_bar": True,
    "progress_bar_refresh_rate": 10,
}

# =============================================================================
# MEDIAPIPE SKELETON CONNECTIONS
# =============================================================================

# Body pose connections (MediaPipe format)
BODY_CONNECTIONS = [
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Eyes
    (9, 10),
    # Mouth
    # Left arm
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Hand connections (MediaPipe format)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]

# =============================================================================
# SAMPLE GLOSS VOCABULARY
# =============================================================================

SAMPLE_GLOSS_VOCABULARY = {
    # Greetings
    "hello": 10,
    "hi": 11,
    "goodbye": 12,
    "thank_you": 13,
    "please": 14,
    "sorry": 15,
    "excuse_me": 16,
    
    # Questions
    "what": 20,
    "where": 21,
    "when": 22,
    "who": 23,
    "why": 24,
    "how": 25,
    "how_much": 26,
    
    # Pronouns
    "i": 30,
    "you": 31,
    "he": 32,
    "she": 33,
    "we": 34,
    "they": 35,
    "me": 36,
    
    # Actions
    "want": 40,
    "need": 41,
    "help": 42,
    "learn": 43,
    "teach": 44,
    "understand": 45,
    "know": 46,
    "think": 47,
    "feel": 48,
    
    # Descriptions
    "good": 50,
    "bad": 51,
    "big": 52,
    "small": 53,
    "fast": 54,
    "slow": 55,
    "hot": 56,
    "cold": 57,
    
    # Time
    "today": 60,
    "tomorrow": 61,
    "yesterday": 62,
    "now": 63,
    "later": 64,
    "morning": 65,
    "night": 66,
    
    # Special tokens
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

# =============================================================================
# COMBINED MODEL CONFIG
# =============================================================================

MODEL_CONFIG = {
    "text_input": TEXT_INPUT_CONFIG,
    "text_to_gloss": TEXT_TO_GLOSS_CONFIG,
    "gloss_to_pose": GLOSS_TO_POSE_CONFIG,
    "refinement": REFINEMENT_CONFIG,
    "output": OUTPUT_CONFIG,
}


def get_config():
    """Get the complete configuration dictionary."""
    return {
        "hardware": HARDWARE_CONFIG,
        "training": TRAINING_CONFIG,
        "model": MODEL_CONFIG,
        "data": DATA_CONFIG,
        "logging": LOGGING_CONFIG,
    }


def print_config():
    """Print current configuration."""
    config = get_config()
    print("=" * 60)
    print("TEXT-TO-SIGN MODEL CONFIGURATION")
    print("=" * 60)
    for section, values in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
