import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define an improved LSTM model
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImprovedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2, num_layers=2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def augment_data(sequence, noise_factor=0.01, time_shift=2):
    """Apply data augmentation to make the model more robust"""
    augmented = sequence.copy()
    
    # Add small random noise
    noise = np.random.normal(0, noise_factor, sequence.shape)
    augmented += noise
    
    # Time shift (circular shift)
    shift = np.random.randint(-time_shift, time_shift + 1)
    if shift != 0:
        augmented = np.roll(augmented, shift, axis=0)
    
    return augmented

def load_and_augment_data():
    """Load data and apply augmentation"""
    print("=== Loading and Augmenting Data ===")
    
    DATA_PATH = os.path.join("MP_Data")
    actions = np.array(["hello", "thanks", "iloveyou"])
    
    # Load and prepare data
    no_sequences = 30
    sequence_length = 30
    label_map = {label: idx for idx, label in enumerate(actions)}
    
    sequences, labels = [], []
    
    for action in actions:
        print(f"\nProcessing {action}...")
        action_sequences = []
        action_labels = []
        
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(
                    os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + ".npy")
                )
                window.append(res)
            
            # Validate the sequence
            window_array = np.array(window)
            if window_array.shape != (sequence_length, 1662):
                print(f"  WARNING: Sequence {sequence} has wrong shape: {window_array.shape}")
                continue
                
            if np.any(np.isnan(window_array)):
                print(f"  WARNING: Sequence {sequence} contains NaN values")
                continue
                
            if np.all(window_array == 0):
                print(f"  WARNING: Sequence {sequence} is all zeros")
                continue
            
            # Add original sequence
            action_sequences.append(window)
            action_labels.append(label_map[action])
            
            # Add augmented versions (especially for hello to help with sequence 0)
            if action == "hello":
                # Create more augmented versions for hello to improve robustness
                for _ in range(3):  # 3 augmented versions per hello sequence
                    augmented = augment_data(window_array)
                    action_sequences.append(augmented)
                    action_labels.append(label_map[action])
            else:
                # Add 1 augmented version for other classes
                augmented = augment_data(window_array)
                action_sequences.append(augmented)
                action_labels.append(label_map[action])
        
        sequences.extend(action_sequences)
        labels.extend(action_labels)
        
        print(f"  Loaded {len(action_sequences)} sequences (including augmented)")
    
    X = np.array(sequences)
    y = np.eye(len(actions), dtype=int)[labels]
    
    print(f"\nTotal data loaded: {X.shape}")
    print(f"Class distribution: {np.sum(y, axis=0)}")
    
    return X, y, actions

def train_improved_model():
    """Train an improved model with data augmentation"""
    print("=== Training Improved Model ===")
    
    # Load data
    X, y, actions = load_and_augment_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize improved model
    input_size = 1662
    hidden_size = 128
    model = ImprovedLSTMModel(input_size=input_size, 
                             hidden_size=hidden_size, 
                             num_classes=actions.shape[0])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # Training loop
    num_epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            targets = torch.argmax(batch_y, dim=1)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                targets = torch.argmax(batch_y, dim=1)
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_classes': actions.shape[0]
                },
                'actions': actions
            }, "improved_sign_language_model.pth")
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Final evaluation
    print("\n=== Final Model Evaluation ===")
    
    # Load best model
    checkpoint = torch.load("improved_sign_language_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on validation set
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            targets = torch.argmax(batch_y, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=actions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_predictions)
    print(cm)
    
    # Test specifically on the problematic hello sequence
    print("\n=== Testing Problematic Hello Sequence ===")
    try:
        # Load the problematic sequence 0
        window = []
        for frame_num in range(30):
            res = np.load(f"MP_Data/hello/0/{frame_num}.npy")
            window.append(res)
        sequence_data = np.array(window)
        
        predicted_sign, confidence, probabilities = predict_sign(model, actions, sequence_data)
        
        print(f"Hello sequence 0:")
        print(f"  Predicted: {predicted_sign}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  All probabilities: {probabilities}")
        
        # Test with slight augmentation
        augmented = augment_data(sequence_data, noise_factor=0.005)
        predicted_sign_aug, confidence_aug, probabilities_aug = predict_sign(model, actions, augmented)
        
        print(f"Hello sequence 0 (augmented):")
        print(f"  Predicted: {predicted_sign_aug}")
        print(f"  Confidence: {confidence_aug:.4f}")
        print(f"  All probabilities: {probabilities_aug}")
        
    except Exception as e:
        print(f"Error testing problematic sequence: {e}")
    
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

if __name__ == "__main__":
    model, actions = train_improved_model()
    print("\nTraining completed! Model saved as 'improved_sign_language_model.pth'") 