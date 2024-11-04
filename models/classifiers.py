import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

class TransformerClassifier(nn.Module):
    '''
    A classifier that distinguishes between real and fake images using a ViT-like transformer.
    '''
    def __init__(self, input_size: int = 512, n_heads: int = 4, hidden_dim: int = 1024, num_encoder_layers: int = 2):
        '''
        Args:
            input_size (int) : the size of the input embedding
            n_heads (int) : the number of heads in the attention
            hidden_dim (int) : the size of the hidden layer of the module
            num_encoder_layers (int) : the number of encoder layers in the nn.TransformerEncoder
        '''
        super(TransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        self.fc = nn.Sequential(nn.Linear(in_features=input_size, out_features=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(0)
        x = F.normalize(x, p=2, dim=1, eps=1e-6)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.fc(x)
        return x

    def train_model(self, train_loader: DataLoader, num_epochs: int, learning_rate: float = 1e-4, device: str = 'cuda', log_interval: int = 500):
        '''
        Train the TransformerClassifier with the provided data.

        Args:
            train_loader (DataLoader): The DataLoader for training data.
            num_epochs (int): The number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device for computation ('cuda' or 'cpu').
            log_interval (int): Number of iterations after which to log the loss.
        '''
        # Move model to specified device
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            total_loss = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device).float()

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(images).squeeze()  # Match BCELoss input

                # Loss computation
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Print loss every log_interval batches
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

        print("Training completed.")


class MLPClassifier(nn.Module):
    '''
    A robust Multi-Layer Perceptron (MLP) classifier designed for binary classification of images as real or fake.
    This model takes a 512-dimensional input and outputs a binary classification.
    '''

    def __init__(self, input_size: int = 512, hidden_dims: list = [512, 256, 128], dropout_rate: float = 0.3):
        '''
        Initializes the MLPClassifier with additional hidden layers.

        Args:
            input_size (int): Dimension of the input features (default: 512).
            hidden_dims (list): List containing dimensions of each hidden layer (default: [512, 256, 128]).
            dropout_rate (float): Dropout rate applied after each hidden layer to prevent overfitting (default: 0.3).
        '''
        super(MLPClassifier, self).__init__()

        # Define the MLP layers
        layers = []
        in_features = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim

        # Add the final output layer for binary classification (2 classes)
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        '''
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with dimension (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor with dimension (batch_size, 2), representing class scores for each sample.
        '''
        return self.model(x)

    def train_model(self, train_loader: DataLoader, num_epochs: int, learning_rate: float = 1e-4, device: str = 'cuda',
                    log_interval: int = 500):
        '''
        Train the MLPClassifier with the provided data.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device for computation ('cuda' or 'cpu').
            log_interval (int): Number of iterations after which to log the loss.
        '''
        # Move model to specified device
        self.to(device)
        criterion = nn.CrossEntropyLoss()  # Cross-entropy for binary classification with 2 output neurons
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            total_loss = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device).long()

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(images)  # No need to squeeze with CrossEntropyLoss

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Print loss every log_interval batches
                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

        print("Training completed.")

