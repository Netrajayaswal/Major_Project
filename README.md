<<<<<<< HEAD
# Text to Sign Language Skeleton Video

This project is a starter pipeline for text-to-sign language translation on a laptop GPU with about 4 GB VRAM. It follows the four stages you described:

1. Text sentence to gloss sequence with `google/mt5-small`.
2. Gloss sequence to 2D MediaPipe-style keypoint frames with a lightweight transformer.
3. Keypoint frames to a black-background skeleton video using OpenCV.
4. Dataset preparation from raw videos or already-rendered skeleton videos.

The important foundation is labeling. You can label manually with `data/labels/train_labels.csv`:

```csv
video_filename,gloss
MVI_9726.mp4,WATER
MVI_9727.mp4,HELLO
```

Use at least 50 unique gloss clips before training, 200+ for better results, and 1000+ for production quality.

You can also label automatically from folders. The folder name becomes the gloss, and leading numbering is removed:

```text
data/skeleton_videos/train/Adjectives/10. Mean/MVI_0001_skeleton.mp4  -> MEAN
data/skeleton_videos/train/Adjectives/79. short/video.mp4            -> SHORT
data/raw_videos/train/1. Dog/MVI_0001.mp4                            -> DOG
```

## Install

Use Python 3.10.
=======



# AI-Powered Sign Language Translation System with Real time Gesture Recognition

### Project Overview

An AI-based assistive technology designed to translate sign language gestures into text and text into sign animations in real time. The system helps bridge the communication gap between deaf and hearing individuals using computer vision and deep learning.

The project uses hand landmark detection and machine learning models to recognize gestures from live webcam input and convert them into readable text.

### Objectives

- Detect hand landmarks using MediaPipe to extract 21 key points from live webcam input.

- Develop a sign-to-text gesture recognition system using deep learning.

- Create a text-to-sign animation generator to display sign language through 3D avatars.

- Build a searchable dictionary containing 500–1000 common sign gestures with video demonstrations.

- Design an interactive learning module with practice mode and real-time feedback.

- Implement user authentication and progress tracking for personalized learning.



## Text-to-Sign Translation Model

A complete system for converting text input into skeleton-based sign language animation videos. Optimized for low-resource hardware (4GB VRAM + 16GB RAM) with checkpoint-based training for overnight training sessions.

##  Features

- **5-Stage Pipeline**: Text → Gloss → Pose → Refinement → Skeleton Video
- **Memory Optimized**: Works on 4GB VRAM with gradient accumulation
- **Night Training**: Automatic training schedule (10 PM - 5 AM)
- **Auto-Resume**: Checkpoints every 30 minutes with automatic resumption
- **Mixed Precision**: FP16 training for memory efficiency
- **Multi-format Output**: NPY skeleton data, MP4 video, optional GIF

## 📁 Project Structure

```
sign_language_project/
├── configs/
│   └── config.py           # All configuration settings
├── models/
│   └── text_to_sign_model.py  # 5-stage model architecture
├── data/
│   ├── dataset.py          # Memory-efficient dataset
│   └── organized_classes/  # Your data goes here
├── utils/
│   ├── checkpoint.py       # Checkpoint & resume system
│   ├── training.py         # Training loop with mixed precision
│   ├── inference.py        # Inference pipeline
│   ├── visualization.py    # Skeleton video rendering
│   └── helpers.py          # Utility functions
├── train.py                # Main training script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

##  Quick Start

### 1. Install Dependencies
>>>>>>> upstream/main

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
## Folder Layout

- `data/raw_videos/train`: original signer videos for training.
- `data/raw_videos/test`: original signer videos for testing.
- `data/skeleton_videos/train`: already-rendered red-dot skeleton videos.
- `data/skeleton_videos/test`: already-rendered red-dot skeleton videos.
- `data/keypoints/train`: extracted training JSON keypoint clips.
- `data/keypoints/valid`: extracted validation JSON keypoint clips.
- `data/keypoints/test`: extracted testing JSON keypoint clips.
- `data/text_gloss/train.csv`: sentence to gloss training pairs.
- `data/text_gloss/valid.csv`: sentence to gloss validation pairs.
- `outputs/checkpoints`: trained model checkpoints.
- `outputs/logs`: training metrics CSV files.
- `outputs/videos`: generated sign skeleton videos.

Nested folders are supported. Put videos inside gloss-named leaf folders, even if there is a category folder above them.

## Setup Labels from Video Folders

For your current folder/subfolder dataset, run:

```bash
python main.py setup-data --videos-dir data/skeleton_videos/train
```

This creates:

- `data/labels/train_labels.csv`
- `data/text_gloss/train.csv`
- `data/text_gloss/valid.csv`

If you change folders or add new videos, run `setup-data` again.

## Prepare Keypoint Data

For best quality, use raw videos and MediaPipe:

```bash
python main.py prepare --mode raw --split train
python main.py prepare --mode raw --split test
```

If you only have rendered skeleton videos with red keypoint dots:

```bash
python main.py prepare --mode rendered --split train --auto-label-from-folder --clean-output
python main.py prepare --mode rendered --split test
```

Rendered-video extraction uses red thresholding and centroid tracking. This is useful for bootstrapping, but raw-video MediaPipe extraction is more reliable because rendered red dots do not always preserve exact anatomical joint identity. For rendered skeleton datasets, `run --pose-source auto` now prefers the original rendered source clips when available, so output does not become random-looking skeleton lines from a bad/old model.

Optional augmentation:

```bash
python scripts/augment_keypoints.py --input-dir data/keypoints/train --output-dir data/keypoints/train --copies 3
```

## Train Stage 1: Text to Gloss

Fill `data/text_gloss/train.csv` and `data/text_gloss/valid.csv` with:

```csv
sentence,gloss_sequence
Hello how are you,HELLO HOW YOU
I need water,I NEED WATER
```

Then train:

```bash
python main.py train-text
```

If your environment has incompatible `peft/bitsandbytes` versions (for example errors around `torch.library.impl_abstract`), the trainer now disables PEFT integrations by default. You can re-enable them only when your package versions are compatible:

```bash
python scripts/train_text2gloss.py --allow-peft
```

The trainer writes BLEU and exact-match metrics to `outputs/logs` and saves the model to `outputs/checkpoints/text2gloss`.
If training stops early and only `checkpoint-*` subfolders exist, the app now auto-detects the latest checkpoint for inference.

## Train Stage 2: Gloss to Pose

After keypoint JSON files exist:

```bash
python main.py train-pose
```

The trainer logs epoch, loss, MPJPE, and keypoint accuracy to `outputs/logs/pose_train_*.csv`. It saves the best checkpoint to `outputs/checkpoints/pose_transformer.pt`.

`train-pose` filters keypoint JSON files through `data/labels/train_labels.csv` by default, so stale keypoints from older experiments do not get mixed into the new model. Use `--no-label-filter` only if you intentionally want to train on every JSON clip in `data/keypoints`.

## Run the Web UI

Start the two-panel translator:

```bash
python main.py
```

This opens a local browser panel at `http://127.0.0.1:7860`:

- Left panel: type English text or a sentence.
- Right panel: plays one skeleton sign video for the entered text.
- Real-time mode: renders automatically after you stop typing for a moment.
- Unknown words: handled by remote sign translation fallback (`sign.mt`) when enabled.
- Remote outage: automatically falls back to local trained+fingerspelling so a playable video is still produced.
- Browser playback: the UI creates `.webm` copies in `outputs/videos/ui` so the right panel can play reliably in Chrome/Edge.

You can also start it explicitly:

```bash
python main.py ui
```

Useful options:

```bash
python main.py ui --no-browser
python main.py ui --port 8080
python main.py ui --pose-source clips
python main.py ui --fallback remote
python main.py ui --fallback pretrained
```

Generated UI videos are saved in `outputs/videos/ui`.

The UI defaults to remote fallback mode so arbitrary text can still produce a skeleton video:

```bash
python main.py ui --fallback remote --remote-spoken en --remote-signed ase
```

If you want fully local trained-only behavior:

```bash
python main.py ui --fallback strict
```

For a local pretrained fallback (optional, large checkpoint):

```bash
python main.py download-pretrained
python main.py ui --fallback pretrained
```

If the browser still shows an old error after code changes, stop the running server with `Ctrl+C`, then run `python main.py` again so the latest UI code loads.

## Run the CLI System

Interactive mode:

```bash
python main.py run
```

Then type a sentence. The output MP4 is written to `outputs/videos`.

Direct text:

```bash
python main.py run --text "Hello how are you"
```

By default, `run` now uses remote fallback mode for arbitrary text:

```bash
python main.py run --text "The robot is dancing on Mars" --fallback remote --remote-spoken en --remote-signed ase
```

If you want fully local trained-only behavior, use `--fallback strict`. If the text model is missing but a pose checkpoint exists, the app uses the trained gloss vocabulary and `data/text_gloss/*.csv` to match known words. For example, if the pose checkpoint only contains `HOME`, then:

```bash
python main.py run --text "I am going home"
```

will generate a real trained `HOME` skeleton video, while:

```bash
python main.py run --text "Hello how are you"
```

will stop and tell you that those words are not trained (strict mode).

For demo-only fingerspelling fallback:

```bash
python main.py run --text "Hello how are you" --fallback fingerspell
```

To render exact training keypoint clips instead of the transformer prediction:

```bash
python main.py run --text "I am going home" --pose-source clips
```

With rendered skeleton video datasets, `--pose-source auto` will also use source clip retrieval when that is the safest output. Use `--pose-source model` only when you specifically want to inspect the transformer-generated pose.

### Sentence Behavior

In strict local mode, the system signs words/glosses that are present in the trained vocabulary or extracted keypoint clips. Any missing important word prevents video generation. Example:

```bash
python main.py run --text "The tall and wide building is clean"
```

If `TALL`, `WIDE`, and `CLEAN` are trained but `BUILDING` is not trained, strict mode reports `BUILDING` as missing and does not play a partial/random video. To make arbitrary sentences accurate locally, add labeled videos/text-gloss pairs for the missing words and retrain, or use remote/pretrained fallback.

Remote mode (`--fallback remote`) follows the `translate-master` API flow and can generate pose output for text outside your local dataset, then renders it into your local skeleton video format. If the remote API fails for a sentence, the pipeline now falls back to local trained signs plus fingerspelling instead of returning an empty/broken video.

Only use preview mode when you want to test the renderer flow:

```bash
python main.py run --text "Hello how are you" --allow-preview
```

That preview is not real sign output.

## Check Project Status

```bash
python main.py status
```

This prints counts for labels, videos, extracted keypoints, text-gloss pairs, checkpoints, and generated videos.

It also prints the trained pose glosses found in `outputs/checkpoints/pose_transformer.pt`. If it says only `HOME`, then the system can only produce `HOME` until you add and train more labeled gloss clips.

## Metrics

- Stage 1: BLEU on gloss sequences, target above `0.4`.
- Stage 2: MPJPE in normalized coordinates, target below `0.05`.
- Stage 2 extra: keypoint accuracy is PCK at threshold `0.05`.
- End to end: native signer review is the most reliable measure.
=======
### 2. Setup Sample Data

```bash
python train.py --setup_sample_data --num_sample_classes 20
```

### 3. Train the Model

```bash
# Start training
python train.py --epochs 100 --batch_size 2

# Resume from checkpoint
python train.py --resume
```

### 4. Run Inference

```bash
# Single text translation
python inference.py --text "Hello, how are you?"

# Interactive mode
python inference.py --interactive

# Batch mode
python inference.py --batch texts.txt outputs/
```

##  Model Architecture

### Stage 1: Text Encoder
- Lightweight transformer encoder
- Multilingual support (English, Hindi, Spanish)
- DistilBERT-style architecture for memory efficiency

### Stage 2: Text-to-Gloss Translation
- Seq2Seq transformer
- Handles sign language grammar differences
- Beam search for inference

### Stage 3: Gloss-to-Pose Generation
- Motion transformer
- Generates 543 keypoints (body, hands, face)
- Temporal attention for smooth motion

### Stage 4: Pose Refinement
- Spatio-Temporal Graph Convolutional Network (ST-GCN)
- Applies kinematic constraints
- Temporal smoothing

### Stage 5: Skeleton Video Output
- OpenCV-based rendering
- MediaPipe skeleton structure
- MP4 video output

## ⚙️ Configuration

Key training parameters in `configs/config.py`:

```python
TRAINING_CONFIG = {
    "batch_size": 2,                    # Small batch for 4GB VRAM
    "gradient_accumulation_steps": 8,   # Effective batch = 16
    "mixed_precision": True,            # FP16 training
    "max_frames": 60,                   # 2.4 seconds at 25fps
    
    # Checkpointing
    "checkpoint_frequency": 30,         # Every 30 minutes
    "training_start_time": "22:00",     # 10 PM
    "training_end_time": "05:00",       # 5 AM
}
```

##  Training Schedule

The system is designed for overnight training:

| Time | Action |
|------|--------|
| 10:00 PM | Training starts automatically |
| Every 30 min | Checkpoint saved |
| 4:55 AM | Graceful shutdown, final checkpoint saved |
| Next night | Auto-resume from last checkpoint |

##  Data Format

### Directory Structure
```
data/organized_classes/
├── class_001_hello/
│   ├── regular/
│   │   ├── video_001.npy
│   │   └── video_002.npy
│   ├── augmented/
│   │   └── aug_video_001.npy
│   └── metadata.json
├── class_002_thank_you/
│   └── ...
```

### Metadata Format
```json
{
  "class_name": "hello",
  "class_id": 1,
  "text_labels": ["hello", "hi", "greetings"],
  "gloss": "HELLO",
  "videos": [
    {
      "filename": "video_001.npy",
      "duration_frames": 60,
      "fps": 25
    }
  ]
}
```

### Skeleton Data Format
- Shape: `[frames, 543, 3]`
- 543 keypoints: 33 body + 21×2 hands + 468 face
- 3 coordinates: x, y, z (normalized 0-1)

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |
| Python | 3.8+ | 3.10+ |

##  Output

The system generates:

1. **Skeleton Data (.npy)**: Raw keypoint coordinates
2. **Video (.mp4)**: Animated skeleton visualization
3. **Optional GIF**: For quick preview

Example usage:
```python
from utils.inference import TextToSignInference
from utils.visualization import create_skeleton_video

# Initialize
inference = TextToSignInference(model_path="checkpoints/best_model.pth")

# Translate
result = inference.translate("Hello, how are you?")

# Save video
create_skeleton_video(result['skeleton'], "output.mp4")
```

##  Training Tips

1. **Start Small**: Use 20 classes, 10 videos each for initial training
2. **Monitor VRAM**: Check `torch.cuda.memory_allocated()` during training
3. **Gradient Accumulation**: Increase `gradient_accumulation_steps` for stability
4. **Early Stopping**: Patience of 15 epochs prevents overfitting
5. **Learning Rate**: Default 3e-4 works well; reduce if unstable

##  Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`

### Slow Training
- Reduce `num_workers` in DataLoader
- Use smaller `max_frames`
- Disable augmentation

### Poor Quality Output
- Train for more epochs
- Increase model capacity
- Add more training data

## 📄 License

This project is for educational and research purposes.

##  Acknowledgments

- MediaPipe for skeleton structure
- PyTorch for deep learning framework
- OpenCV for video processing
>>>>>>> 7dfeda3 (Initial code)
>>>>>>> upstream/main
