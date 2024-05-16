import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(178, 16)
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, seq_len):
        super(MyCNN, self).__init__()
        self.seq_len = seq_len
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Max Pooling Layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 * (((self.seq_len - 4) // 2 - 4) // 2), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=5)  # Output layer
        
    def forward(self, x):
        # Convolutional Layer 1 followed by ReLU activation and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        
        # Convolutional Layer 2 followed by ReLU activation and max pooling
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16 * (((self.seq_len - 4) // 2 - 4) // 2))
        
        # Fully Connected Layer 1 followed by ReLU activation
        x = self.relu(self.fc1(x))
        
        # Output Layer
        x = self.fc2(x)
        
        return x










import torch.nn.functional as F

class MyRNN(nn.Module):
    def __init__(self, l2_reg=0.01):
        super(MyRNN, self).__init__()

        # Define the GRU layer
        self.gru = nn.GRU(178, 32, 1, batch_first=True)  # Increased hidden size

        # Define additional fully connected layer
        self.fc1 = nn.Linear(32, 16)  # Additional fully connected layer
        self.fc2 = nn.Linear(16, 5)   # Output layer

        # Define dropout layers
        self.dropout = nn.Dropout(0.2)

        # Define batch normalization layers
        self.batch_norm = nn.BatchNorm1d(178)  # Apply batch norm before input to GRU

        # Define the CNN part
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 6, 5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(6, 16, 5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            torch.nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(656, 178),
        )

        # L2 regularization (weight decay)
        self.l2_reg = l2_reg

    def forward(self, x):
        # Apply batch normalization
        x = self.batch_norm(x)

        # Permute dimensions for GRU input
        x = x.permute(0, 2, 1)

        # Pass through CNN
        x = self.cnn(x).unsqueeze(1)

        # Pass through GRU
        out, _ = self.gru(x)

        # Apply dropout
        out = self.dropout(out)

        # Apply fully connected layers
        out = F.relu(self.fc1(out[:, -1]))  # Pass only the last output of GRU
        out = self.fc2(out)

        return out














class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()

        # First fully-connected layer (embedding layer)
        self.fc1 = nn.Linear(dim_input, 32)
        self.tanh = nn.Tanh()

        # GRU layer
        self.gru = nn.GRU(input_size=32, hidden_size=16, batch_first=True)

        # Output layer
        self.fc2 = nn.Linear(16, 2)  # Two output units for binary classification

        # Reduce dropout rate
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_tuple):
        seqs, lengths = input_tuple

        # Apply the first fully-connected layer (embedding layer) with tanh activation
        x = self.fc1(seqs)
        x = self.tanh(x)

        # Apply dropout
        x = self.dropout(x)

        # Pack the padded sequences
        packed_seqs = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Apply the GRU layer
        output, _ = self.gru(packed_seqs)

        # Unpack the sequences
        unpacked_seqs, _ = pad_packed_sequence(output, batch_first=True)

        # Apply the output layer
        x = self.fc2(unpacked_seqs[:, -1, :])

        return x





