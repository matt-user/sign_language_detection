import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define a simpler LSTM model
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

def load_and_validate_data():
    """Load data and perform validation checks"""
    print("=== Loading and Validating Data ===")
    
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
            
            action_sequences.append(window)
            action_labels.append(label_map[action])
        
        sequences.extend(action_sequences)
        labels.extend(action_labels)
        
        print(f"  Loaded {len(action_sequences)} valid sequences")
    
    X = np.array(sequences)
    y = np.eye(len(actions), dtype=int)[labels]
    
    print(f"\nTotal data loaded: {X.shape}")
    print(f"Class distribution: {np.sum(y, axis=0)}")
    
    return X, y, actions

def train_simple_model():
    """Train a simpler model with better regularization"""
    print("=== Training Simple Model ===")
    
    # Load data
    X, y, actions = load_and_validate_data()
    
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize simpler model
    input_size = 1662
    hidden_size = 128
    model = SimpleLSTMModel(input_size=input_size, 
                           hidden_size=hidden_size, 
                           num_classes=actions.shape[0])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # Training loop
    num_epochs = 500
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
            }, "simple_sign_language_model.pth")
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
    checkpoint = torch.load("simple_sign_language_model.pth", weights_only=False)
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
    
    # Test with individual class examples
    print("\n=== Testing Individual Class Examples ===")
    for i, action in enumerate(actions):
        # Find examples of this class in test set
        class_indices = [j for j, target in enumerate(all_targets) if target == i]
        if class_indices:
            # Test first example
            example_idx = class_indices[0]
            example_X = X_test[example_idx:example_idx+1]
            example_tensor = torch.FloatTensor(example_X)
            
            with torch.no_grad():
                output = model(example_tensor)
                probabilities = output.squeeze().numpy()
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
            
            print(f"{action} example:")
            print(f"  Predicted: {actions[predicted_class]}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  All probabilities: {probabilities}")
    
    return model, actions

if __name__ == "__main__":
    model, actions = train_simple_model()
    print("\nTraining completed! Model saved as 'simple_sign_language_model.pth'") 