import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import mediapipe as mp
from pytorch_model import LSTMModel
from utils import extract_keypoints, mp_holistic, mp_drawing

def load_model(model_path="improved_sign_language_model.pth"):
    """Load the trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    # Load the checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    actions = checkpoint['actions']
    
    # Create model instance based on model type
    if 'hidden_size1' in model_config:
        # Complex model (original)
        model = LSTMModel(
            input_size=model_config['input_size'],
            hidden_size1=model_config['hidden_size1'],
            hidden_size2=model_config['hidden_size2'],
            hidden_size3=model_config['hidden_size3'],
            num_classes=model_config['num_classes']
        )
    elif 'fc1.weight' in checkpoint['model_state_dict']:
        # Improved model (new) - has fc1 and fc2 layers
        from improved_model import ImprovedLSTMModel
        model = ImprovedLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_classes=model_config['num_classes']
        )
    else:
        # Simple model - has single fc layer
        from simple_model import SimpleLSTMModel
        model = SimpleLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_classes=model_config['num_classes']
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training info: Epoch {checkpoint['epoch']}, Final Loss: {checkpoint['loss']:.4f}")
    print(f"Actions: {actions}")
    
    return model, actions



def predict_sign(model, actions, sequence_data):
    """Predict sign language from sequence data"""
    model.eval()
    with torch.no_grad():
        # Ensure sequence_data is a numpy array
        if isinstance(sequence_data, list):
            sequence_data = np.array(sequence_data)
        
        # Ensure correct shape (30, 1662)
        if sequence_data.shape != (30, 1662):
            print(f"Warning: Expected shape (30, 1662), got {sequence_data.shape}")
            return "unknown", 0.0, np.zeros(len(actions))
        
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
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Sequence for prediction (30 frames)
    sequence = []
    threshold = 0.8  # Lower confidence threshold for better detection
    
    # Class-specific thresholds (based on analysis)
    # class_thresholds = {
    #     "hello": 0.8,      # Very reliable
    #     "thanks": 0.6,     # Some ambiguity with iloveyou
    #     "iloveyou": 0.5    # Higher threshold due to similarity with thanks
    # }
    
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
                
                # Get class-specific threshold
                # class_threshold = class_thresholds.get(predicted_sign, threshold)
                class_threshold = 0.5
                
                # Display predictionq
                if confidence > class_threshold:
                    # Color based on confidence
                    if confidence > 0.9:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.8:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    cv2.putText(image, f'Prediction: {predicted_sign}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Show all probabilities
                    for i, (action, prob) in enumerate(zip(actions, probabilities)):
                        y_pos = 110 + i * 30
                        prob_color = (255, 255, 255)  # White
                        if action == predicted_sign:
                            prob_color = color  # Use prediction color
                        cv2.putText(image, f'{action}: {prob:.3f}', (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 1)
                else:
                    cv2.putText(image, f'Low confidence: {predicted_sign}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f'Try again (need {class_threshold:.1f})', (10, 70), 
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