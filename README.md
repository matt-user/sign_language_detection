# Sign Language Detection Model

This project contains a PyTorch-based LSTM model for real-time sign language detection using MediaPipe keypoints.

## Essential Files

### Core Files
- `improved_model.py` - Improved LSTM model definition and training script with data augmentation
- `load_model.py` - Model loading and real-time prediction
- `improved_sign_language_model.pth` - Trained model weights (12.7MB)
- `requirements_pytorch.txt` - Python dependencies

### Data Collection
- `main.py` - Data collection script for capturing sign language gestures
- `utils.py` - Utility functions for MediaPipe detection and keypoint extraction

### Data
- `MP_Data/` - Training data directory containing keypoint sequences for:
  - `hello/` - Hello gesture data (30 sequences)
  - `thanks/` - Thanks gesture data (30 sequences)
  - `iloveyou/` - I love you gesture data (30 sequences)

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
python improved_model.py
```

### 3. Run Real-time Prediction
```bash
python load_model.py
```

## Model Architecture

The improved model uses a robust LSTM-based architecture with data augmentation:

### Architecture Details
- **Input**: 30 frames × 1662 keypoints (pose + face + hands)
- **LSTM Layers**: 2-layer LSTM with 128 hidden units and dropout (0.2)
- **Fully Connected**: 128 → 64 → 3 units with ReLU activation
- **Regularization**: Dropout (0.3) at multiple layers
- **Output**: 3 classes (hello, thanks, iloveyou) with softmax

### Training Features
- **Data Augmentation**: Noise addition and time shifting
- **Class Balancing**: 3x augmentation for "hello" class
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Validation**: 20% test split with stratification

### Performance
- **Training Accuracy**: 100%
- **Validation Accuracy**: 100%
- **Test Accuracy**: 100% on all classes
- **Model Size**: 12.7MB

## Supported Gestures

- **Hello** - Wave gesture
- **Thanks** - Thank you gesture
- **I Love You** - ILY sign gesture

## Usage

### Data Collection (Optional)
1. Run `python main.py` to collect training data
2. Follow the on-screen instructions to record gestures
3. Each gesture requires 30 sequences of 30 frames each

### Training
1. Run `python improved_model.py` to train the model
2. Training includes data augmentation and validation
3. Best model is automatically saved as `improved_sign_language_model.pth`

### Real-time Prediction
1. Run `python load_model.py` for webcam prediction
2. Make sign language gestures in front of the camera
3. Ensure good lighting and clear hand visibility
4. Press 'q' to quit

### Model Loading
```python
from load_model import load_model, predict_sign

# Load the trained model
model, actions = load_model()

# Make predictions
predicted_sign, confidence, probabilities = predict_sign(model, actions, sequence_data)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- TensorBoard (for training logs)

See `requirements_pytorch.txt` for exact versions.

## Troubleshooting

### Real-time Prediction Issues
- **Ensure good lighting** for MediaPipe detection
- **Position hands clearly** in front of the camera
- **Use exact same gestures** as training data
- **Check hand visibility** - hands must be fully visible

### Model Performance
- The model achieves 100% accuracy on training data
- Real-time performance depends on webcam conditions
- Lower confidence threshold if needed for real-time use

## Project Structure
```
sign_language_detection/
├── improved_model.py                    # Training script
├── improved_sign_language_model.pth     # Model file
├── load_model.py                        # Model loading & prediction
├── main.py                              # Data collection
├── utils.py                             # Utilities
├── requirements_pytorch.txt             # Dependencies
├── README.md                           # This file
└── MP_Data/                            # Training data
    ├── hello/                          # Hello gestures
    ├── thanks/                         # Thanks gestures
    └── iloveyou/                       # I love you gestures
``` 