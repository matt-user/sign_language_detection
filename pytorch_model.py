import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc1 = nn.Linear(hidden_size3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM layers - PyTorch LSTM returns (output, (hidden, cell))
        # For return_sequences=True, we use the full output sequence
        # For return_sequences=False, we use only the last output
        lstm_out, _ = self.lstm1(x)  # Returns full sequence
        lstm_out, _ = self.lstm2(lstm_out)  # Returns full sequence
        lstm_out, (hidden, cell) = self.lstm3(lstm_out)  # Returns full sequence, but we only need the last output
        
        # Take only the last output from the final LSTM layer (equivalent to return_sequences=False)
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size3)
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        
        return out

def train_model():
    """Train the sign language detection model"""
    # Data loading and preprocessing
    DATA_PATH = os.path.join("MP_Data")
    actions = np.array(["hello", "thanks", "iloveyou"])

    # Load and prepare data (same as in main.py)
    no_sequences = 30
    sequence_length = 30
    label_map = {label: idx for idx, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(
                    os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + ".npy")
                )
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = np.eye(len(actions), dtype=int)[labels]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_size = 1662  # Based on the original input_shape=(30,1662)
    model = LSTMModel(input_size=input_size, 
                      hidden_size1=64, 
                      hidden_size2=128, 
                      hidden_size3=64, 
                      num_classes=actions.shape[0])
     
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # TensorBoard setup
    log_dir = os.path.join('Logs')
    writer = SummaryWriter(log_dir)

    # Training loop
    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            
            # Convert one-hot encoded targets to class indices for CrossEntropyLoss
            targets = torch.argmax(batch_y, dim=1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Log to TensorBoard
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', accuracy, epoch)
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Example prediction (equivalent to the original res = [.7, 0.2, 0.1])
    model.eval()
    with torch.no_grad():
        # Create a dummy input for demonstration
        dummy_input = torch.randn(1, 30, 1662)  # batch_size=1, sequence_length=30, input_size=1662
        prediction = model(dummy_input)
        predicted_class = torch.argmax(prediction, dim=1)
        print(f"Predicted class: {actions[predicted_class.item()]}")

    # Print model summary
    print("\nModel Summary:")
    print(model)

    # Save the trained model
    model_save_path = "sign_language_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'loss': avg_loss,
        'model_config': {
            'input_size': input_size,
            'hidden_size1': 64,
            'hidden_size2': 128,
            'hidden_size3': 64,
            'num_classes': actions.shape[0]
        },
        'actions': actions
    }, model_save_path, _use_new_zipfile_serialization=False)

    print(f"Model saved to {model_save_path}")

    # Close TensorBoard writer
    writer.close()

    print("Training completed!")
    
    return model, actions

if __name__ == "__main__":
    train_model() 