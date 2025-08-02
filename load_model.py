import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import mediapipe as mp
from pytorch_model import LSTMModel

def load_model(model_path="sign_language_model.pth"):
    """Load the trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    # Load the checkpoint with weights_only=False for compatibility
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=False, trying alternative method...")
        # Alternative loading method for newer PyTorch versions
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    actions = checkpoint['actions']
    
    # Create model instance
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_size1=model_config['hidden_size1'],
        hidden_size2=model_config['hidden_size2'],
        hidden_size3=model_config['hidden_size3'],
        num_classes=model_config['num_classes']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training info: Epoch {checkpoint['epoch']}, Final Loss: {checkpoint['loss']:.4f}")
    print(f"Actions: {actions}")
    
    return model, actions

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    left_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, left_hand, right_hand])

def predict_sign(model, actions, sequence_data):
    """Predict sign language from sequence data"""
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)  # Shape: (1, 30, 1662)
        
        # Get prediction
        output = model(input_tensor)
        probabilities = output.squeeze().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return actions[predicted_class], confidence, probabilities

def real_time_prediction():
    """Real-time sign language prediction using webcam"""
    # Load model
    model, actions = load_model()
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Sequence for prediction (30 frames)
    sequence = []
    threshold = 0.7  # Confidence threshold
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            
            # Keep only last 30 frames
            if len(sequence) > 30:
                sequence = sequence[-30:]
            
            # Make prediction when we have enough frames
            if len(sequence) == 30:
                predicted_sign, confidence, probabilities = predict_sign(model, actions, sequence)
                
                # Display prediction
                if confidence > threshold:
                    cv2.putText(image, f'Prediction: {predicted_sign}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, 'No sign detected', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Sign Language Detection', image)
            
            # Break on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    try:
        # Load model
        model, actions = load_model()
        
        # Example prediction with dummy data
        print("\nExample prediction with dummy data:")
        dummy_sequence = np.random.randn(30, 1662)  # 30 frames, 1662 features
        predicted_sign, confidence, probabilities = predict_sign(model, actions, dummy_sequence)
        print(f"Predicted: {predicted_sign}")
        print(f"Confidence: {confidence:.2f}")
        print(f"All probabilities: {probabilities}")
        
        print("\nStarting real-time prediction...")
        print("Press 'q' to quit the webcam feed")
        real_time_prediction()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run pytorch_model.py first to train and save the model.") 