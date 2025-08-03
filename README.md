# Sign Language Detection Model

This project contains a PyTorch-based LSTM model for real-time sign language detection using MediaPipe keypoints.

## Essential Files

### Core Files
- `main.py` - Data preparation and training pipeline
- `pytorch_model.py` - LSTM model definition and training script
- `load_model.py` - Model loading and real-time prediction
- `sign_language_model.pth` - Trained model weights
- `requirements_pytorch.txt` - Python dependencies

### Data
- `MP_Data/` - Training data directory containing keypoint sequences for:
  - `hello/` - Hello gesture data
  - `thanks/` - Thanks gesture data  
  - `iloveyou/` - I love you gesture data

### Environment
- `env/` - Virtual environment with all dependencies

## Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements_pytorch.txt
```

### 2. Train the Model
```bash
python pytorch_model.py
```

### 3. Run Real-time Prediction
```bash
python load_model.py
```

## Model Architecture

The model uses an LSTM-based architecture:
- Input: 30 frames × 1662 keypoints (pose + face + hands)
- Hidden layers: 64 → 128 → 64 units
- Output: 3 classes (hello, thanks, iloveyou)

## Supported Gestures

- **Hello** - Wave gesture
- **Thanks** - Thank you gesture  
- **I Love You** - ILY sign gesture

## Usage

1. **Training**: Run `python pytorch_model.py` to train the model on the MP_Data
2. **Prediction**: Run `python load_model.py` for real-time webcam prediction
3. **Model Loading**: Use `load_model()` function to load the trained model in other scripts

## Requirements

- Python 3.8+
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- TensorBoard (for training logs)

See `requirements_pytorch.txt` for exact versions. 